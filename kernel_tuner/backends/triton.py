import logging
import numpy as np

from kernel_tuner.backends.backend import GPUBackend

try:
    import torch
except ImportError:
    logging.error("Torch not available")

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None
    logging.error("Unable to load triton")


class TritonFunctions(GPUBackend):

    def __init__(self, device=0, iterations=7, compiler_options=None, observers=None):
        if not triton or not torch:
            logging.error("Triton or torch not available")
            raise ImportError("Triton or torch not available")

        self.stream = torch.cuda.default_stream()
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.stop_event = torch.cuda.Event(enable_timing=True)

        super().__init__(device=device, iterations=iterations, compiler_options=compiler_options, observers=observers)

    def ready_argument_list(self, arguments):
        # Allocate memory here
        print("Triton ready args list")
        pass

    def compile(self, kernel_instance):
        logging.debug("Compiling triton kernel")
        return triton.jit(kernel_instance)

    def start_event(self):
        logging.debug("Start triton event")
        self.start_event.record()

    def stop_event(self):
        logging.debug("Stop triton event")
        self.stop_event.record()

    def kernel_finished(self):
        logging.debug("Checking if kernel has finished")
        return self.stop_event.query()

    def run_kernel(self, func, gpu_args, threads, grid, stream):
        # Run the kernel
        if stream is None:
            stream = self.stream



        logging.debug("Running triton kernel")
        pass

    def synchronize(self):
        torch.cuda.synchronize()

    def memset(self, allocation, value, size):
        pass

    def memcpy_dtoh(self, dest, src):
        pass

    def memcpy_htod(self, dest, src):
        pass

    def copy_constant_memory_args(self, cmem_args):
        pass

    def copy_shared_memory_args(self, smem_args):
        pass

    def copy_texture_memory_args(self, texmem_args):
        pass




