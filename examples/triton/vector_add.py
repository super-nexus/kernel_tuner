import numpy
import triton.language as tl
from kernel_tuner import tune_kernel, run_kernel
from kernel_tuner.file_utils import store_output_file, store_metadata_file


def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # note: `constexpr` so it can be used as a shape value.
               ):
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


size = 10000000

a = numpy.random.randn(size).astype(numpy.float32)
b = numpy.random.randn(size).astype(numpy.float32)
c = numpy.zeros_like(b)
n = numpy.int32(size)

args = [c, a, b, n]

tune_params = dict()
# tune_params["block_size_x"] = [2**i for i in range(10)] THIS IS ONLY NEEDED FOR TUNING
tune_params["block_size_x"] = 256

results = run_kernel(
    kernel_name="add_kernel",
    kernel_source=add_kernel,
    problem_size=size,
    arguments=args,
    params=tune_params,
    lang="triton"
)

print(results)
