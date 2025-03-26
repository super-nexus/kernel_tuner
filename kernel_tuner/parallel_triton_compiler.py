"""
Parallel Triton Compiler Module

This module provides functionality to compile Triton kernels in parallel
across multiple CPU cores, significantly speeding up the tuning process.
"""

import os
import time
import logging
import concurrent.futures
import multiprocessing as mp
import json
from pathlib import Path
from typing import Dict, List, Callable, Any, Tuple, Optional, Union

import torch
import numpy as np
import triton
from triton.backends.compiler import GPUTarget

from kernel_tuner.interface import Options
from kernel_tuner.searchspace import Searchspace
from kernel_tuner.kernel_sources.kernel_source_factory import get_kernel_source
from kernel_tuner.backends.triton import TritonFunctions

class ParallelTritonCompiler:
    """
    A class that compiles Triton kernels in parallel across multiple CPU threads.
    
    This class is designed to be used as a preprocessing step before tuning,
    to compile all kernel configurations in parallel and cache the results.
    
    Note: This uses ThreadPoolExecutor rather than ProcessPoolExecutor to avoid
    pickling issues with Triton kernel functions. This means compilation will be
    limited by Python's Global Interpreter Lock (GIL), but Triton compilation is
    mostly C++ code which releases the GIL, so parallelization should still be effective.
    """
    
    def __init__(
        self, 
        kernel_fn: Callable, 
        kernel_name: str,
        arguments: List[any],
        max_workers: int = None,
        cache_dir: str = None,
        verbose: bool = True
    ):
        """
        Initialize the parallel compiler.
        
        Args:
            kernel_fn: The Triton kernel function to compile
            arguments: The arguments of the kernel function
            max_workers: Maximum number of worker processes to use (defaults to os.cpu_count())
            cache_dir: Directory to cache compiled kernels
            verbose: Whether to print verbose output
        """
        self.kernel_fn = kernel_fn
        self.kernel_name = kernel_name
        self.arguments = arguments
        self.max_workers = max_workers or os.cpu_count()
        self.verbose = verbose

        # Set up cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None
            
        self.compilation_results = {}
        self.cache_file = self._get_cache_file() if self.cache_dir else None
        self.cache_data = self._load_cache() if self.cache_file else {}
        
    def _get_cache_file(self) -> Path:
        """Get the path to the cache file for this kernel."""
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{self.kernel_name}_cache.json"
    
    def _config_to_hash(self, config: Dict[str, Any]) -> str:
        """Convert a configuration to a unique hash string."""
        import hashlib
        
        # Handle both dictionaries and tuples
        if isinstance(config, dict):
            # Sort the keys to ensure consistent hashing
            sorted_items = sorted(config.items())
            # Convert to a string and hash
            config_str = json.dumps(sorted_items)
        elif isinstance(config, tuple):
            # For tuples, just convert to string directly
            config_str = str(config)
        else:
            raise TypeError(f"Unsupported config type: {type(config)}")
            
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _load_cache(self) -> Dict:
        """Load the cache file if it exists."""
        if not self.cache_file or not self.cache_file.exists():
            return {}
        
        try:
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load cache file: {str(e)}")
            return {}
    
    def _save_cache(self):
        """Save the current cache data to file."""
        if not self.cache_file:
            return
            
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache_data, f)
        except Exception as e:
            print(f"Warning: Failed to save cache file: {str(e)}")
    
    def _compile_single_config(self, config: Union[Dict[str, Any], Tuple]) -> Tuple[Union[Dict[str, Any], Tuple], bool, Dict]:
        """
        Compile a single kernel configuration.
        
        Args:
            config: The kernel configuration to compile (dict or tuple)
            
        Returns:
            Tuple of (config, success, cache_data)
        """
        config_hash = self._config_to_hash(config)

        # Check if already in cache
        if config_hash in self.cache_data and self.cache_data[config_hash].get('success', False):
            print(f"Config {config_hash} found in cache")
            return config, True, None
        
        try:
            # Convert tuple to dict if needed for compilation
            constants = config

            kernel_source = get_kernel_source(self.kernel_name, self.kernel_fn, lang='TRITON', defines=None)
            updated_fn, tmp_file_path = kernel_source.apply_params_to_source_fn(config)

            if isinstance(config, tuple):
                # We need param names to convert tuple to dict
                # This is a limitation - we can't compile tuples directly
                print(f"Warning: Tuple config received but can't be compiled directly. Skipping {config}")
                return config, False, None
            
            # Compile using warmup
            kernel_jit = triton.jit(updated_fn)

            start_time = time.time()
            kernel_jit.warmup(grid=(1, 1, 1), *self.arguments, **constants)
            compile_time = time.time() - start_time

            # Prepare cache data to be saved in main process
            cache_data = {
                'config': str(config) if isinstance(config, tuple) else config,
                'compile_time': compile_time,
                'success': True
            }
            
            print(f"Successfully compiled config {config_hash} in {compile_time:.2f}s")
            # Delete the temporary file
            if os.path.exists(tmp_file_path):
                try:
                    os.remove(tmp_file_path)
                except Exception as e:
                    print(f"Warning: Failed to delete temporary file {tmp_file_path}: {str(e)}")

            return config, True, cache_data
        except Exception as e:
            print(f"Failed to compile config {config_hash}: {str(e)}")
            
            # Prepare cache data for failure
            cache_data = {
                'config': str(config) if isinstance(config, tuple) else config,
                'error': str(e),
                'success': False
            }
            
            return config, False, cache_data
    
    def _compile_single_config_wrapper(self, config: Union[Dict[str, Any], Tuple]) -> Tuple[Union[Dict[str, Any], Tuple], bool, Dict]:
        """
        Wrapper function to catch any unexpected exceptions during compilation.
        
        Args:
            config: The kernel configuration to compile
            
        Returns:
            Tuple of (config, success, cache_data)
        """
        try:
            return self._compile_single_config(config)
        except Exception as e:
            config_hash = self._config_to_hash(config)
            
            # Log the failure
            print(f"Caught unexpected exception while compiling {config_hash}: {type(e).__name__}: {str(e)}")
            
            # Prepare cache data for failure
            cache_data = {
                'config': str(config) if isinstance(config, tuple) else config,
                'error': f"{type(e).__name__}: {str(e)}",
                'success': False
            }
            
            return config, False, cache_data
    
    def _scan_configs(self, configs: List[Union[Dict[str, Any], Tuple]]) -> Tuple[List, List]:
        """
        Scan configurations to determine which ones are cached.
        
        Args:
            configs: List of configurations to check
            
        Returns:
            Tuple of (cached_configs, uncached_configs)
        """
        cached_configs = []
        uncached_configs = []
        
        print("Scanning cache for existing configurations...")
        start_time = time.time()
        
        for config in configs:
            config_hash = self._config_to_hash(config)
            if config_hash in self.cache_data:
                cached_configs.append(config)
            else:
                uncached_configs.append(config)
                
            # Print progress periodically
            total_processed = len(cached_configs) + len(uncached_configs)
            if self.verbose or total_processed % max(1, len(configs) // 10) == 0:
                print(f"Cache scan progress: {total_processed}/{len(configs)} "
                      f"({total_processed/len(configs)*100:.1f}%)")
        
        scan_time = time.time() - start_time
        print(f"Cache scan complete in {scan_time:.2f}s:")
        print(f"- Found {len(cached_configs)} cached configurations")
        print(f"- Need to compile {len(uncached_configs)} new configurations")
        
        return cached_configs, uncached_configs

    def compile_configs(self, configs: List[Union[Dict[str, Any], Tuple]]) -> Dict[str, bool]:
        """
        Compile multiple kernel configurations in parallel.
        
        Args:
            configs: List of kernel configurations to compile
            
        Returns:
            Dictionary mapping configuration hashes to compilation success
        """
        results = {}
        start_time = time.time()
        
        # First scan cache for existing configurations in parallel
        cached_configs, configs_to_compile = self._scan_configs(configs)
        
        # Add cached configs to results
        for config in cached_configs:
            config_hash = self._config_to_hash(config)
            results[config_hash] = True
        
        if not configs_to_compile:
            print("All configurations already cached!")
            return results
        
        # Continue with compilation of uncached configs
        print(f"Compiling {len(configs_to_compile)} configurations using {self.max_workers} workers...")
        
        # Process configurations in smaller batches to recover from worker failures
        batch_size = self.max_workers
        for i in range(0, len(configs_to_compile), batch_size):
            batch = configs_to_compile[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(configs_to_compile) + batch_size - 1)//batch_size}")

            # Create a fresh process pool for each batch
            mp_context = mp.get_context('spawn')
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers, mp_context=mp_context) as executor:
                # Submit all compilation tasks
                future_to_config = {}
                for config in batch:
                    future = executor.submit(self._compile_single_config_wrapper, config)
                    future_to_config[future] = config
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_config):
                    config = future_to_config[future]
                    config_hash = self._config_to_hash(config)
                    
                    try:
                        _, success, cache_data = future.result()
                        results[config_hash] = success
                        # Store cache data to be saved after batch
                        if cache_data:
                            self.cache_data[config_hash] = cache_data
                    except concurrent.futures.process.BrokenProcessPool:
                        print(f"Worker process crashed while compiling {config_hash}. Marking as failed.")
                        results[config_hash] = False
                    except Exception as e:
                        print(f"Exception during compilation of {config_hash}: {str(e)}")
                        results[config_hash] = False
                    
                    # Print progress
                    completed = len(results)
                    if self.verbose or completed % max(1, len(configs) // 10) == 0:
                        print(f"Progress: {completed}/{len(configs)} configurations processed ({completed/len(configs)*100:.1f}%)")
            
            # Save cache after each batch is complete
            self._save_cache()
        
        # Print summary
        total_time = time.time() - start_time
        successful = sum(1 for success in results.values() if success)
        
        print(f"Compilation complete: {successful}/{len(configs)} successful in {total_time:.2f}s")
        print(f"Average time per configuration: {total_time/len(configs):.2f}s")
        print(f"Effective configurations per second: {len(configs)/total_time:.2f}")
        
        return results
    
    def compile_searchspace(self, tune_params: Dict[str, List], restrictions: Optional[Union[Callable, List[str]]] = None) -> Dict[str, bool]:
        """
        Compile all valid configurations in a search space.
        
        Args:
            tune_params: Dictionary of parameter names to lists of possible values
            restrictions: Optional restrictions on the search space
            
        Returns:
            Dictionary mapping configuration hashes to compilation success
        """
        # Create a searchspace object
        searchspace = Searchspace(tune_params, restrictions, max_threads=4)
        
        # Generate all valid configurations
        configs = []
        param_names = list(tune_params.keys())
        
        # Convert tuples to dictionaries
        for config_tuple in searchspace.sorted_list():
            config_dict = {param_names[i]: config_tuple[i] for i in range(len(param_names))}
            configs.append(config_dict)
        
        return self.compile_configs(configs)

    def get_successful_configs(self) -> List[Union[Dict[str, Any], Tuple]]:
        """
        Get the list of successfully compiled configurations.
        
        Returns:
            List of successfully compiled configurations
        """
        return [config['config'] for config in self.cache_data.values() if config.get('success', False)]

def parallel_compile_triton_kernel(
    kernel_name: str,
    kernel_fn: Callable,
    tune_params: Dict[str, List],
    arguments: List[any],
    restrictions: Optional[Union[Callable, List[str]]] = None,
    max_workers: int = None,
    cache_dir: str = None,
    verbose: bool = False
) -> Tuple[Dict[str, bool], List[Union[Dict[str, Any], Tuple]]]:
    """
    Compile all valid configurations of a Triton kernel in parallel.
    
    This function distributes the compilation of Triton kernels across multiple CPU cores,
    significantly speeding up the tuning process. It also caches the compiled kernels
    to avoid recompiling the same configuration multiple times.
    
    Args:
        kernel_fn: The Triton kernel function to compile
        tune_params: Dictionary of parameter names to lists of possible values
        restrictions: Optional restrictions on the search space
        max_workers: Maximum number of worker processes to use (defaults to os.cpu_count())
        cache_dir: Directory to cache compiled kernels
        verbose: Whether to print verbose output
        
    Returns:
        Tuple of (dictionary mapping configuration hashes to compilation success,
                 list of successfully compiled configurations)
    """
    compiler = ParallelTritonCompiler(
        kernel_name=kernel_name,
        kernel_fn=kernel_fn,
        arguments=arguments,
        max_workers=max_workers,
        cache_dir=cache_dir,
        verbose=verbose
    )
    
    results = compiler.compile_searchspace(tune_params, restrictions)
    return results, compiler.get_successful_configs() 

def get_already_compiled_configs(
    cache_dir: str,
    kernel_name: str,
) -> List[Union[Dict[str, Any], Tuple]]:
    """
    Get the list of already compiled configurations from the cache directory.
    """
    cache_file = Path(cache_dir) / f"{kernel_name}_cache.json"
    if not cache_file.exists():
        return []
        
    try:
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
            # Return only successful configurations
            return [config['config'] for config in cache_data.values() if config.get('success', False)]
    except Exception as e:
        print(f"Warning: Failed to load cache file: {str(e)}")
        return []