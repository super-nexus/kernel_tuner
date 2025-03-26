"""Triton Brute Force strategy module.

This module implements a brute force strategy specifically for Triton kernels
that works directly with pre-compiled configurations.
"""

import logging
from typing import List, Dict, Any
from time import perf_counter

from kernel_tuner.searchspace import Searchspace

def triton_brute_force(
    raw_configs: List[Dict[str, Any]],
    runner,
    tuning_options: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Brute force strategy that works directly with pre-compiled Triton configurations.
    
    This strategy is designed to work with pre-compiled Triton configurations
    without needing to construct a search space. It simply evaluates each
    configuration in sequence.
    
    Args:
        searchspace: The searchspace object (not used in this strategy)
        runner: The runner object for executing configurations
        tuning_options: Dictionary containing tuning options
        
    Returns:
        List of dictionaries containing the results of all configurations
    """
    # Get the raw configurations from tuning_options
    raw_configs = tuning_options.get('triton_raw_configs')
    if not raw_configs:
        raise ValueError("triton_raw_configs must be provided for triton_brute_force strategy")

    results = runner.run(raw_configs, tuning_options)
           
    return results 