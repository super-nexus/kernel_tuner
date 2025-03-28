"""
Example script demonstrating parallel compilation of Triton kernels.
"""

import os
import time
import argparse
import torch
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget

from kernel_tuner.parallel_triton_compiler import parallel_compile_triton_kernel, get_already_compiled_configs
from kernel_tuner.interface import tune_kernel

# Define a simple matrix multiplication kernel
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension.
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    # Optional meta-parameters
):
    """Matrix multiplication kernel."""
    # Program ID
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Create pointers for the first blocks of A and B
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Iterate to compute a block of the C matrix
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # Compute the matrix multiplication
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Apply activation function
    accumulator = tl.where(accumulator >= 0, accumulator, 0.01 * accumulator)
   
    # Store the result
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def main(m):
    parser = argparse.ArgumentParser(description="Parallel Triton Kernel Compilation Example")
    parser.add_argument("--cache-dir", type=str, default="triton_cache", help="Directory to cache compiled kernels")
    parser.add_argument("--max-workers", type=int, default=None, help="Maximum number of worker processes")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    args = parser.parse_args()


    problem_size = (m, m, 1)
    matrix_size = (problem_size[0], problem_size[1])
    a = torch.randn(matrix_size, dtype=torch.float16, device="cuda")
    b = torch.randn(matrix_size, dtype=torch.float16, device="cuda")
    c = torch.empty(matrix_size, dtype=torch.float16, device="cuda")
    M, K = a.shape
    _, N = b.shape
    M = torch.tensor(M, dtype=torch.int32, device="cuda").item()
    K = torch.tensor(K, dtype=torch.int32, device="cuda").item()
    N = torch.tensor(N, dtype=torch.int32, device="cuda").item()

    stride_am = torch.tensor(a.stride(0), dtype=torch.int32, device="cuda").item()
    stride_ak = torch.tensor(a.stride(1), dtype=torch.int32, device="cuda").item()
    stride_bk = torch.tensor(b.stride(0), dtype=torch.int32, device="cuda").item()
    stride_bn = torch.tensor(b.stride(1), dtype=torch.int32, device="cuda").item()
    stride_cm = torch.tensor(c.stride(0), dtype=torch.int32, device="cuda").item()
    stride_cn = torch.tensor(c.stride(1), dtype=torch.int32, device="cuda").item()

    arguments = [
        a, b, c,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
    ]
    # Define the parameter space
    start_time = time.time()

    tune_params = dict()
    tune_params['BLOCK_SIZE_M'] = [16 * 2 ** i for i in range(6)]
    tune_params['BLOCK_SIZE_N'] = [16 * 2 ** i for i in range(6)]
    tune_params['BLOCK_SIZE_K'] = [16 * 2 ** i for i in range(6)]
    tune_params['num_stages'] = [1, 2, 3, 4, 5]
    tune_params['num_warps'] = [2, 4, 8]
    tune_params['GROUP_SIZE_M'] = [i for i in range(4, 10)]

    results = parallel_compile_triton_kernel(
        kernel_name="matmul_kernel",
        kernel_fn=matmul_kernel,
        arguments=arguments,
        tune_params=tune_params,
        max_workers=args.max_workers,
        cache_dir=args.cache_dir,
        verbose=args.verbose
    )
    total_time = time.time() - start_time
    
    # Print results
    successful = sum(1 for success in results.values() if success)
    print(f"Compilation results: {successful}/{len(results)} configurations compiled successfully")
    print(f"Total time: {total_time:.2f}s")

def tune_matmul(m):
    problem_size = (m, m, 1)
    matrix_size = (problem_size[0], problem_size[1])
    a = torch.randn(matrix_size, dtype=torch.float16)
    b = torch.randn(matrix_size, dtype=torch.float16)
    c = torch.empty(matrix_size, dtype=torch.float16)
    M, K = a.shape
    _, N = b.shape
    M = torch.tensor(M, dtype=torch.int32)
    K = torch.tensor(K, dtype=torch.int32)
    N = torch.tensor(N, dtype=torch.int32)

    stride_am = torch.tensor(a.stride(0), dtype=torch.int32)
    stride_ak = torch.tensor(a.stride(1), dtype=torch.int32)
    stride_bk = torch.tensor(b.stride(0), dtype=torch.int32)
    stride_bn = torch.tensor(b.stride(1), dtype=torch.int32)
    stride_cm = torch.tensor(c.stride(0), dtype=torch.int32)
    stride_cn = torch.tensor(c.stride(1), dtype=torch.int32)

    arguments = [
        a, b, c,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
    ]

    tune_params = dict()
    tune_params['BLOCK_SIZE_M'] = [16 * 2 ** i for i in range(6)]
    tune_params['BLOCK_SIZE_N'] = [16 * 2 ** i for i in range(6)]
    tune_params['BLOCK_SIZE_K'] = [16 * 2 ** i for i in range(6)]
    tune_params['num_stages'] = [1, 2, 3, 4, 5]
    tune_params['num_warps'] = [2, 4, 8]
    tune_params['GROUP_SIZE_M'] = [i for i in range(4, 10)]

    # First compile all configurations in parallel
    print("Getting already compiled configurations...")
    cached_configs = get_already_compiled_configs(
        cache_dir="triton_cache",
        kernel_name="matmul_kernel"
    )
    print(f"Found {len(cached_configs)} successfully compiled configurations")
    
    print("Starting tuning with successfully compiled configurations...")
    results, env = tune_kernel(
        kernel_name='matmul_kernel',
        kernel_source=matmul_kernel,
        problem_size=problem_size,
        arguments=arguments,
        tune_params=tune_params,  # Use only successfully compiled configurations
        lang='TRITON',
        block_size_names=["BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K"],
        triton_raw_configs=cached_configs,
        strategy='triton_brute_force',
        strategy_options={'triton_raw_configs': cached_configs},
    )

    # Filter out failed configurations and format results
    valid_results = []
    for config in results:
        # Check if time is a valid positive float
        try:
            time = float(config['time'])
            if time > 0:
                config['time'] = time  # Ensure it's stored as float
                valid_results.append(config)
        except (ValueError, TypeError):
            continue
    
    return valid_results

if __name__ == "__main__":
    main(16384)