import torch
import triton.language as tl
import numpy as np
import gc
import os
import json
from datetime import datetime

from kernel_tuner.interface import tune_kernel

# Check for required environment variable
cache_dir = os.getenv('KERNEL_TUNER_CACHE_DIR')
cache_file_name = os.getenv('KERNEL_TUNER_CACHE_FILE', 'group_gemm_results.json')

if cache_dir is None:
    raise ValueError("Environment variable KERNEL_TUNER_CACHE_DIR must be set")

cache_file = os.path.join(cache_dir, cache_file_name)

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
    "NUM_SM": [2 ** i for i in range(6, 10)]
}

constraints = [
    "BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z <= 524288"
]

grid_div_x = ["1/NUM_SM"]
grid_div_y = []
grid_div_z = []

def tune_group_gemm(N):
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

    cache_file_name = f'{cache_file}_{N}.json'

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
        cache=cache_file_name,
    )
    
    # Filter out failed configurations and format results
    valid_results = []
    for config in res:
        # Check if time is a valid positive float
        try:
            time = float(config['time'])
            if time > 0:
                config['time'] = time  # Ensure it's stored as float
                valid_results.append(config)
        except (ValueError, TypeError):
            continue
    
    return valid_results

if __name__ == '__main__':
    matrix_sizes = [4096, 8192, 16384, 32768, 65536]
    all_results = {
        "gpu_info": {
            "gpu_name": torch.cuda.get_device_name()
        }
    }
    
    for size in matrix_sizes:
        results = tune_group_gemm(size)
        gc.collect()
        torch.cuda.empty_cache()
        all_results[str(size)] = results
    
    # Add timestamp to filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'group_gemm_results_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
