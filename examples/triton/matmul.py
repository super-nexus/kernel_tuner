import torch
import gc
import os
import json

from triton import language as tl
from kernel_tuner.interface import run_kernel, tune_kernel


def matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_X: tl.constexpr, BLOCK_SIZE_Y: tl.constexpr, BLOCK_SIZE_Z: tl.constexpr,  #
        GROUP_SIZE_M=8,  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_X)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_Y)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)) % M
    offs_bn = (pid_n * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_Z)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_X, BLOCK_SIZE_Y), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_Z)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_Z, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_Z, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_Z * stride_ak
        b_ptrs += BLOCK_SIZE_Z * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    accumulator = tl.where(accumulator >= 0, accumulator, 0.01 * accumulator)

    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    offs_cn = pid_n * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


TORCH_HAS_FP8 = hasattr(torch, "float8_e5m2")
if not TORCH_HAS_FP8 or not torch.cuda.is_available():
    raise RuntimeError("This example requires a GPU with FP8 support.")


# Check for required environment variable
cache_dir = os.getenv('KERNEL_TUNER_CACHE_DIR')
cache_file_name = os.getenv('KERNEL_TUNER_CACHE_FILE', 'matmul_results.json')

if cache_dir is None:
    raise ValueError("Environment variable KERNEL_TUNER_CACHE_DIR must be set")

cache_file = os.path.join(cache_dir, cache_file_name)


def tune_matmul(m):
    problem_size = (m, m, 1)
    matrix_size = (problem_size[0], problem_size[1])
    a = torch.randn(matrix_size, dtype=torch.float16)
    b = torch.randn(matrix_size, dtype=torch.float16)
    a = a.to(torch.float8_e5m2)
    b = b.T
    b = b.to(torch.float8_e5m2)
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
    tune_params['BLOCK_SIZE_X'] = [16 * 2 ** i for i in range(6)]
    tune_params['BLOCK_SIZE_Y'] = [16 * 2 ** i for i in range(6)]
    tune_params['BLOCK_SIZE_Z'] = [16 * 2 ** i for i in range(6)]
    tune_params['num_stages'] = [1, 2, 3, 4, 5]
    tune_params['num_warps'] = [1, 2, 4, 8]
    tune_params['GROUP_SIZE_M'] = [i for i in range(4, 10)]

    cache_file_name = f'{cache_file}_{m}.json'

    results, env = tune_kernel(
        kernel_name='matmul_kernel',
        kernel_source=matmul_kernel,
        problem_size=problem_size,
        arguments=arguments,
        tune_params=tune_params,
        lang='TRITON',
        block_size_names=["BLOCK_SIZE_X", "BLOCK_SIZE_Y", "BLOCK_SIZE_Z"],
        cache=cache_file_name,
        strategy='basinhopping',
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


if __name__ == '__main__':
    mat_sizes = [32768, 65536]
    all_results = {}
    
    # Get GPU information
    gpu_name = torch.cuda.get_device_name()
    gpu_info = {
        "gpu_name": gpu_name,
    }
    all_results["gpu_info"] = gpu_info

    for m in mat_sizes:
        result = tune_matmul(m)
        gc.collect()
        torch.cuda.empty_cache()
        all_results[str(m)] = result

    # Add timestamp to filename
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'matmul_results_{timestamp}.json'

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)