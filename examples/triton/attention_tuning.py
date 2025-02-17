import torch
import gc
import triton
import triton.language as tl
from kernel_tuner.interface import tune_kernel
import numpy as np
import json
import os

# Check for required environment variable
cache_dir = os.getenv('KERNEL_TUNER_CACHE_DIR')
if cache_dir is None:
    raise ValueError("Environment variable KERNEL_TUNER_CACHE_DIR must be set")

cache_file = os.path.join(cache_dir, 'attention_tuning_results.json')

@triton.jit
def _attn_fwd_inner(
    acc, l_i, m_i, q, K_block_ptr, V_block_ptr, mask_block_ptr,
    stride_k_seqlen, stride_v_seqlen, stride_attn_mask_kv_seqlen,
    start_m, qk_scale, q_load_mask,
    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,
    KV_CTX: tl.constexpr, fp8_v: tl.constexpr,
    HAS_ATTN_MASK: tl.constexpr, PRE_LOAD_V: tl.constexpr,
):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
    else:
        lo, hi = 0, KV_CTX

    K_block_ptr += lo * stride_k_seqlen
    V_block_ptr += lo * stride_v_seqlen
    kv_load_mask = lo + offs_n < KV_CTX
    if HAS_ATTN_MASK:
        mask_block_ptr += lo * stride_attn_mask_kv_seqlen

    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        k = tl.load(K_block_ptr, mask=kv_load_mask[None, :], other=0.0)

        if PRE_LOAD_V:
            v = tl.load(V_block_ptr, mask=kv_load_mask[:, None], other=0.0)

        qk = tl.dot(q, k, allow_tf32=False)

        if HAS_ATTN_MASK:
            attn_mask = tl.load(
                mask_block_ptr,
                mask=q_load_mask[:, None] & kv_load_mask[None, :],
                other=0.0,
            )

        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])

            if HAS_ATTN_MASK:
                qk = qk * qk_scale + attn_mask
                qk *= 1.44269504
                qk = qk + tl.where(mask, 0, -1.0e6)
            else:
                qk_scale *= 1.44269504
                qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)

            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            if HAS_ATTN_MASK:
                qk = qk * qk_scale + attn_mask
                qk *= 1.44269504
                qk = qk - m_ij[:, None]
            else:
                qk_scale *= 1.44269504
                qk = qk * qk_scale - m_ij[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        if not PRE_LOAD_V:
            v = tl.load(V_block_ptr, mask=kv_load_mask[:, None], other=0.0)
        if fp8_v:
            p = p.to(tl.float8e5)
        else:
            p = p.to(q.dtype)
        acc = tl.dot(p, v, acc, allow_tf32=False)
        m_i = m_ij

        K_block_ptr += BLOCK_N * stride_k_seqlen
        V_block_ptr += BLOCK_N * stride_v_seqlen

        if HAS_ATTN_MASK:
            mask_block_ptr += BLOCK_N * stride_attn_mask_kv_seqlen

    return acc, l_i, m_i

def attention_kernel(
    Q, K, V, attn_mask, sm_scale, Out,
    stride_q_batch, stride_q_head, stride_q_seqlen, stride_q_headsize,
    stride_k_batch, stride_k_head, stride_k_seqlen, stride_k_headsize,
    stride_v_batch, stride_v_head, stride_v_seqlen, stride_v_headsize,
    stride_attn_mask_batch, stride_attn_mask_head,
    stride_attn_mask_q_seqlen, stride_attn_mask_kv_seqlen,
    stride_o_batch, stride_o_head, stride_o_seqlen, stride_o_headsize,
    Z, q_numhead, kv_numhead, Q_CTX, KV_CTX,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr, HAS_ATTN_MASK: tl.constexpr, PRE_LOAD_V: tl.constexpr
):
    """Implementation of the attention kernel for tuning"""
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    batch_id = off_hz // q_numhead
    head_id = off_hz % q_numhead
    kv_head_id = off_hz % kv_numhead

    q_offset = batch_id * stride_q_batch + head_id * stride_q_head
    kv_offset = batch_id * stride_k_batch + kv_head_id * stride_k_head

    offs_headsize = tl.arange(0, HEAD_DIM)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    q_load_mask = offs_m < Q_CTX
    offs_n = tl.arange(0, BLOCK_N)

    Q_block_ptr = (
        Q + q_offset + offs_m[:, None] * stride_q_seqlen
        + offs_headsize[None, :] * stride_q_headsize
    )
    K_block_ptr = (
        K + kv_offset + offs_n[None, :] * stride_k_seqlen
        + offs_headsize[:, None] * stride_k_headsize
    )
    V_block_ptr = (
        V + kv_offset + offs_n[:, None] * stride_v_seqlen
        + offs_headsize[None, :] * stride_v_headsize
    )

    if HAS_ATTN_MASK:
        attn_mask_offset = (
            batch_id * stride_attn_mask_batch
            + head_id * stride_attn_mask_head
        )
        mask_block_ptr = (
            attn_mask + attn_mask_offset
            + offs_m[:, None] * stride_attn_mask_q_seqlen
            + offs_n[None, :] * stride_attn_mask_kv_seqlen
        )
    else:
        mask_block_ptr = None

    O_block_ptr = (
        Out + q_offset + offs_m[:, None] * stride_o_seqlen
        + offs_headsize[None, :] * stride_o_headsize
    )

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    qk_scale = sm_scale
    q = tl.load(Q_block_ptr, mask=q_load_mask[:, None], other=0.0)

    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q, K_block_ptr, V_block_ptr, mask_block_ptr,
            stride_k_seqlen, stride_v_seqlen, stride_attn_mask_kv_seqlen,
            start_m, qk_scale, q_load_mask, BLOCK_M, HEAD_DIM, BLOCK_N,
            4 - STAGE, offs_m, offs_n, KV_CTX, False, HAS_ATTN_MASK, PRE_LOAD_V,
        )

    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q, K_block_ptr, V_block_ptr, mask_block_ptr,
            stride_k_seqlen, stride_v_seqlen, stride_attn_mask_kv_seqlen,
            start_m, qk_scale, q_load_mask, BLOCK_M, HEAD_DIM, BLOCK_N,
            2, offs_m, offs_n, KV_CTX, False, HAS_ATTN_MASK, PRE_LOAD_V,
        )

    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask=q_load_mask[:, None])

def tune_attention(batch_size=2, seq_len=128, head_dim=64, num_heads=4):
    # Create sample inputs (torch.Tensor)
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
    key = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
    value = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
    out = torch.empty_like(query)
    
    # Create a dummy attention mask (all ones)
    attn_mask = torch.ones((batch_size, num_heads, seq_len, seq_len), device='cuda')
    
    # Scale factor (numpy scalar)
    sm_scale = np.float32(1.0 / (head_dim ** 0.5))

    HEAD_DIM_Q, HEAD_DIM_K = query.shape[-1], key.shape[-1]
    HEAD_DIM_V = value.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}

    
    # Convert all scalar integers to numpy integers
    arguments = [
        query, key, value, attn_mask, sm_scale, out,
        np.int32(query.stride(0)), np.int32(query.stride(1)), 
        np.int32(query.stride(2)), np.int32(query.stride(3)),
        np.int32(key.stride(0)), np.int32(key.stride(1)), 
        np.int32(key.stride(2)), np.int32(key.stride(3)),
        np.int32(value.stride(0)), np.int32(value.stride(1)), 
        np.int32(value.stride(2)), np.int32(value.stride(3)),
        np.int32(attn_mask.stride(0)), np.int32(attn_mask.stride(1)), 
        np.int32(attn_mask.stride(2)), np.int32(attn_mask.stride(3)), 
        np.int32(out.stride(0)), np.int32(out.stride(1)), 
        np.int32(out.stride(2)), np.int32(out.stride(3)),
        np.int32(batch_size * num_heads),  # Z
        np.int32(num_heads),  # q_numhead
        np.int32(num_heads),  # kv_numhead
        np.int32(seq_len),    # Q_CTX
        np.int32(seq_len),    # KV_CTX
  ]

    # Start with a single configuration for testing
    tune_params = {
        'HEAD_DIM': [HEAD_DIM_K],
        'HAS_ATTN_MASK': [False],
        'STAGE': [1],
        'BLOCK_M': [16, 32, 64, 128, 256, 512, 1024],
        'BLOCK_N': [16, 32, 64, 128, 256, 512, 1024],
        'PRE_LOAD_V': [True, False],
        'num_stages': [1, 2, 3, 4],
        'num_warps': [1, 2, 4, 8],
    }

    # Simple constraint
    constraints = [
        "BLOCK_N <= HEAD_DIM"
    ]

    problem_size = (
        query.shape[2],
        query.shape[0] * query.shape[1],
        1,
    )

    grid_div_x = ["BLOCK_M"]
    grid_div_y = ["1"]
    grid_div_z = ["1"]

    results, env = tune_kernel(
        kernel_name='attention_kernel',
        kernel_source=attention_kernel,
        problem_size=problem_size,
        arguments=arguments,
        tune_params=tune_params,
        restrictions=constraints,
        lang='TRITON',
        grid_div_x=grid_div_x,
        grid_div_y=grid_div_y,
        grid_div_z=grid_div_z,
        block_size_names=['BLOCK_M', 'BLOCK_N'],
        strategy='genetic_algorithm',
        strategy_options={
            'maxiter': 10000
        },
        cache=cache_file,
    )

    return results

if __name__ == '__main__':
    # Test with a single configuration first
    results = tune_attention()
    
    # Filter out failed compilations and find best config
    valid_results = [result for result in results if isinstance(result['time'], (int, float))]
    if valid_results:
        best_config = min(valid_results, key=lambda x: x['time'])
        print("\nBest configuration:")
        print(json.dumps(best_config, indent=2))
    else:
        print("\nNo valid configurations found - all compilations failed") 