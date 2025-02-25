import os
os.environ["TRITON_ALWAYS_COMPILE"] = "1"

import torch
import triton.language as tl
import numpy as np
import gc
import time
import pandas as pd

from kernel_tuner.interface import tune_kernel


def grouped_matmul_kernel(
    # device tensor of matrices pointers
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    # device tensor of gemm sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <M, N, K> of each gemm
    group_gemm_sizes,
    # device tensor of leading dimension sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <lda, ldb, ldc> of each gemm
    g_lds,
    # number of gemms
    group_size,
    # number of virtual SM
    NUM_SM: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
    BLOCK_SIZE_Z: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    for g in range(group_size):
        # get the gemm size of the current problem
        gm = tl.load(group_gemm_sizes + g * 3)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)
        gk = tl.load(group_gemm_sizes + g * 3 + 2)
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_X)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_Y)
        num_tiles = num_m_tiles * num_n_tiles
        # iterate through the tiles in the current gemm problem
        while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
            # pick up a tile from the current gemm problem
            k = gk
            lda = tl.load(g_lds + g * 3)
            ldb = tl.load(g_lds + g * 3 + 1)
            ldc = tl.load(g_lds + g * 3 + 2)
            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.float16))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.float16))
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.float16))
            # figure out tile coordinates
            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m_idx = tile_idx_in_gemm // num_n_tiles
            tile_n_idx = tile_idx_in_gemm % num_n_tiles

            # do regular gemm here
            offs_am = tile_m_idx * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
            offs_bn = tile_n_idx * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
            offs_k = tl.arange(0, BLOCK_SIZE_Z)
            
            # Add masks for boundary checking
            mask_m = offs_am < gm
            mask_n = offs_bn < gn
            mask_k = offs_k < k
            
            a_ptrs = a_ptr + offs_am[:, None] * lda + offs_k[None, :]
            b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_bn[None, :]
            accumulator = tl.zeros((BLOCK_SIZE_X, BLOCK_SIZE_Y), dtype=tl.float32)
            
            for kk in range(0, tl.cdiv(k, BLOCK_SIZE_Z)):
                k_remaining = k - kk * BLOCK_SIZE_Z
                k_mask = offs_k < k_remaining
                
                # Load with masks
                a = tl.load(a_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0)
                b = tl.load(b_ptrs, mask=k_mask[:, None] & mask_n[None, :], other=0.0)
                
                # hint to Triton compiler to do proper loop pipelining
                tl.multiple_of(a_ptrs, [16, 16])
                tl.multiple_of(b_ptrs, [16, 16])
                accumulator += tl.dot(a, b)
                a_ptrs += BLOCK_SIZE_Z
                b_ptrs += BLOCK_SIZE_Z * ldb
            c = accumulator.to(tl.float16)

            offs_cm = tile_m_idx * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
            offs_cn = tile_n_idx * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
            c_ptrs = c_ptr + ldc * offs_cm[:, None] + offs_cn[None, :]

            # Store with masks
            tl.store(c_ptrs, c, mask=mask_m[:, None] & mask_n[None, :])

            # go to the next tile by advancing NUM_SM
            tile_idx += NUM_SM

        # get ready to go to the next gemm problem
        last_problem_end = last_problem_end + num_tiles


# Constants and parameters that don't change
tunable_params = {
    "BLOCK_SIZE_X": [2 ** i for i in range(4, 9)],
    "BLOCK_SIZE_Y": [2 ** i for i in range(4, 9)],
    "BLOCK_SIZE_Z": [2 ** i for i in range(4, 9)],
    "NUM_SM": [60, 72, 82, 90, 105]
}

constraints = [
    "BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z <= 524288"
]

grid_div_x = ["1/NUM_SM"]
grid_div_y = []
grid_div_z = []

def tune_group_gemm(N, strategy="brute_force"):
    group_size = np.int32(4)
    group_A = []
    group_B = []
    A_addrs = []
    B_addrs = []
    C_addrs = []
    g_sizes = []
    g_lds = []
    group_C = []
    for i in range(group_size):
        A = torch.rand((N, N), device="cuda", dtype=torch.float16)
        B = torch.rand((N, N), device="cuda", dtype=torch.float16)
        C = torch.empty((N, N), device="cuda", dtype=torch.float16)
        group_A.append(A)
        group_B.append(B)
        group_C.append(C)
        A_addrs.append(A.data_ptr())
        B_addrs.append(B.data_ptr())
        C_addrs.append(C.data_ptr())
        g_sizes += [N, N, N]
        g_lds += [N, N, N]

    d_a_ptrs = torch.tensor(A_addrs, device="cuda")
    d_b_ptrs = torch.tensor(B_addrs, device="cuda")
    d_c_ptrs = torch.tensor(C_addrs, device="cuda")
    d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device="cuda")
    d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device="cuda")

    problem_size = (1, 1, 1)

    args = [
        d_a_ptrs,
        d_b_ptrs,
        d_c_ptrs,
        d_g_sizes,
        d_g_lds,
        group_size,
    ]

    start_time = time.time()
    res, env = tune_kernel(
        kernel_name="grouped_matmul_kernel",
        kernel_source=grouped_matmul_kernel,
        problem_size=problem_size,
        arguments=args,
        tune_params=tunable_params,
        grid_div_x=grid_div_x,
        grid_div_y=grid_div_y,
        grid_div_z=grid_div_z,
        restrictions=constraints,
        lang="TRITON",
        block_size_names=["BLOCK_SIZE_X", "BLOCK_SIZE_Y", "BLOCK_SIZE_Z"],
        strategy=strategy
    )
    end_time = time.time()
    
    return res, end_time - start_time

def get_min_valid_time(results):
    valid_times = [result['time'] for result in results if isinstance(result['time'], (int, float))]
    return min(valid_times) if valid_times else float('inf')

def load_existing_results(force=False):
    """Load existing results from CSV file if it exists."""
    if os.path.exists('strategy_comparison_results.csv') and not force:
        return pd.read_csv('strategy_comparison_results.csv')
    return pd.DataFrame()

def get_brute_force_baseline(existing_results, matrix_size, force=False):
    """Get or compute brute force baseline performance."""
    if not force and not existing_results.empty and \
       len(existing_results[existing_results['Strategy'] == 'brute_force']) > 0:
        print("Loading existing brute force baseline...")
        brute_force_row = existing_results[existing_results['Strategy'] == 'brute_force'].iloc[0]
        return brute_force_row['Performance (ms)'], None
    
    print("Running brute force baseline...")
    brute_force_results, brute_time = tune_group_gemm(matrix_size, "brute_force")
    best_performance = get_min_valid_time(brute_force_results)
    
    best_config = next(
        result
        for result in brute_force_results 
        if isinstance(result['time'], (int, float)) and result['time'] == best_performance
    )
    
    baseline_result = {
        'Strategy': 'brute_force',
        'Matrix Size': matrix_size,
        'Tuning Time (s)': brute_time,
        'Performance (ms)': best_performance,
        'Performance Ratio': 1.0,
        'Config': best_config
    }
    
    return best_performance, baseline_result

def should_skip_strategy(strategy, size, existing_results):
    """Check if strategy should be skipped."""
    return not existing_results.empty and \
           len(existing_results[(existing_results['Strategy'] == strategy) & 
                              (existing_results['Matrix Size'] == size)]) > 0

def run_strategy(strategy, size, best_performance):
    """Run a single strategy and return its results."""
    results, tuning_time = tune_group_gemm(size, strategy)
    performance = get_min_valid_time(results)
    performance_ratio = performance / best_performance if performance != float('inf') else float('inf')
    
    best_config = next(
        result
        for result in results 
        if isinstance(result['time'], (int, float)) and result['time'] == performance
    )
    
    return {
        'Strategy': strategy,
        'Matrix Size': size,
        'Tuning Time (s)': tuning_time,
        'Performance (ms)': performance,
        'Performance Ratio': performance_ratio,
        'Config': best_config
    }

if __name__ == '__main__':
    strategies = [
        "basinhopping", "diff_evo", "firefly_algorithm", "genetic_algorithm", "greedy_ils",
        "greedy_mls", "minimize", "mls", "ordered_greedy_mls", "pso",
        "random_sample"
    ]
    
    matrix_sizes = [8192]
    results_data = []
    force = False
    update_baseline_only = True  # New flag to control baseline-only updates
    
    # Load existing results
    existing_results = load_existing_results(force)
    
    # Get baseline performance
    best_performance, baseline_result = get_brute_force_baseline(existing_results, matrix_sizes[0], True)  # Force baseline update
    
    if baseline_result:
        # If we have existing results, update only the baseline
        if not existing_results.empty and update_baseline_only:
            # Update or add the baseline result
            existing_results = existing_results[existing_results['Strategy'] != 'brute_force']
            existing_results = pd.concat([pd.DataFrame([baseline_result]), existing_results])
            
            # Update performance ratios for all other strategies
            existing_results['Performance Ratio'] = existing_results.apply(
                lambda row: row['Performance (ms)'] / best_performance 
                if row['Strategy'] != 'brute_force' else 1.0, 
                axis=1
            )
            
            combined_df = existing_results
        else:
            # Regular flow for new results
            results_data.append(baseline_result)
            
            # Test each strategy
            for strategy in strategies:
                print(f"Testing strategy: {strategy}")
                for size in matrix_sizes:
                    if not force and should_skip_strategy(strategy, size, existing_results):
                        print(f"Skipping {strategy} for size {size} - already in results")
                        continue
                        
                    try:
                        result = run_strategy(strategy, size, best_performance)
                        results_data.append(result)
                        
                        gc.collect()
                        torch.cuda.empty_cache()
                        
                    except Exception as e:
                        print(f"Error with strategy {strategy}: {str(e)}")
    
    # Convert results to DataFrame
    new_results_df = pd.DataFrame(results_data)
    
    # Merge with existing results
    if not existing_results.empty:
        # Concatenate new results with existing ones and remove duplicates
        combined_df = pd.concat([existing_results, new_results_df]).drop_duplicates(
            subset=['Strategy', 'Matrix Size'], 
            keep='last'
        )
    else:
        combined_df = new_results_df
    
    print("\nResults Summary:")
    print(combined_df)
    
    # Save combined results to CSV
    combined_df.to_csv('strategy_comparison_results.csv', index=False)
