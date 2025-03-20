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
        
    def _get_cache_path(self, config_hash: str) -> Path:
        """Get the path to the cache file for a given configuration hash."""
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{config_hash}.json"
    
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
    
    def _compile_single_config(self, config: Union[Dict[str, Any], Tuple]) -> Tuple[Union[Dict[str, Any], Tuple], bool]:
        """
        Compile a single kernel configuration.
        
        Args:
            config: The kernel configuration to compile (dict or tuple)
            
        Returns:
            Tuple of (config, success)
        """
        config_hash = self._config_to_hash(config)
        cache_path = self._get_cache_path(config_hash)

        # Check if already in cache
        if cache_path and cache_path.exists():
            print(f"Config {config_hash} found in cache")
            return config, True
        
        try:
            # Convert tuple to dict if needed for compilation
            constants = config

            kernel_source = get_kernel_source(self.kernel_name, self.kernel_fn, lang='TRITON', defines=None)
            updated_fn, tmp_file_path = kernel_source.apply_params_to_source_fn(config)

            if isinstance(config, tuple):
                # We need param names to convert tuple to dict
                # This is a limitation - we can't compile tuples directly
                print(f"Warning: Tuple config received but can't be compiled directly. Skipping {config}")
                return config, False
            
            # Compile using warmup
            kernel_jit = triton.jit(updated_fn)

            start_time = time.time()
            kernel_jit.warmup(grid=(1, 1, 1), *self.arguments, **constants)
            compile_time = time.time() - start_time

            # Cache the result if caching is enabled
            if cache_path:
                with open(cache_path, 'w') as f:
                    json.dump({
                        'config': str(config) if isinstance(config, tuple) else config,
                        'compile_time': compile_time,
                        'success': True
                    }, f)
            
            
            print(f"Successfully compiled config {config_hash} in {compile_time:.2f}s")
            # Delete the temporary file
            if os.path.exists(tmp_file_path):
                try:
                    os.remove(tmp_file_path)
                except Exception as e:
                    print(f"Warning: Failed to delete temporary file {tmp_file_path}: {str(e)}")

            return config, True
        except Exception as e:
            print(f"Failed to compile config {config_hash}: {str(e)}")
            
            # Cache the failure if caching is enabled
            if cache_path:
                with open(cache_path, 'w') as f:
                    json.dump({
                        'config': str(config) if isinstance(config, tuple) else config,
                        'error': str(e),
                        'success': False
                    }, f)
            
            return config, False
    
    def _compile_single_config_wrapper(self, config: Union[Dict[str, Any], Tuple]) -> Tuple[Union[Dict[str, Any], Tuple], bool]:
        """
        Wrapper function to catch any unexpected exceptions during compilation.
        
        Args:
            config: The kernel configuration to compile
            
        Returns:
            Tuple of (config, success)
        """
        try:
            return self._compile_single_config(config)
        except Exception as e:
            config_hash = self._config_to_hash(config)
            cache_path = self._get_cache_path(config_hash)
            
            # Log and cache the failure
            print(f"Caught unexpected exception while compiling {config_hash}: {type(e).__name__}: {str(e)}")
            
            if cache_path:
                try:
                    with open(cache_path, 'w') as f:
                        json.dump({
                            'config': str(config) if isinstance(config, tuple) else config,
                            'error': f"{type(e).__name__}: {str(e)}",
                            'success': False
                        }, f)
                except:
                    print(f"Failed to write to cache file for {config_hash}")
            
            return config, False
    
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
        
        print(f"Compiling {len(configs)} configurations using {self.max_workers} workers...")
        # Print the first config for debug
        first_config = configs[0]
        first_config_hash = self._config_to_hash(first_config)
        print(f"First config to compile: {first_config} (hash: {first_config_hash})")

        # Process configurations in smaller batches to recover from worker failures
        batch_size = 32  # Adjust based on your needs
        for i in range(0, len(configs), batch_size):
            batch = configs[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(configs) + batch_size - 1)//batch_size}")

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
                        _, success = future.result()
                        results[config_hash] = success
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


def parallel_compile_triton_kernel(
    kernel_name: str,
    kernel_fn: Callable,
    tune_params: Dict[str, List],
    arguments: List[any],
    restrictions: Optional[Union[Callable, List[str]]] = None,
    max_workers: int = None,
    cache_dir: str = None,
    verbose: bool = False
) -> Dict[str, bool]:
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
        Dictionary mapping configuration hashes to compilation success
        
    Example:
        ```python
        @triton.jit
        def my_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            # Kernel code here
            pass
            
        signature = {0: "*fp32", 1: "*fp32", 2: "i32"}
        tune_params = {"BLOCK_SIZE": [32, 64, 128, 256], "num_warps": [1, 2, 4, 8]}
        
        results = parallel_compile_triton_kernel(
            kernel_fn=my_kernel,
            tune_params=tune_params,
            signature=signature,
            cache_dir="triton_cache"
        )
        ```
    """
    compiler = ParallelTritonCompiler(
        kernel_name=kernel_name,
        kernel_fn=kernel_fn,
        arguments=arguments,
        max_workers=max_workers,
        cache_dir=cache_dir,
        verbose=verbose
    )
    
    return compiler.compile_searchspace(tune_params, restrictions) 