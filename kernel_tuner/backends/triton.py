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
    logging.error("Unable to load triton")


class TritonFunctions(GPUBackend):

    def __init__(self, device=0, iterations=7, compiler_options=None, observers=None):
        # Do something here
        super().__init__(device=device, iterations=iterations, compiler_options=compiler_options, observers=observers)
        pass

    def ready_argument_list(self, arguments):
        # Allocate memory here
        pass

    def compile(self, kernel_instance):
        # Not sure if this is needed
        logging.debug("Compiling triton kernel")
        pass

    def start_event(self):
        logging.debug("Starting triton event")
        pass

    def stop_event(self):
        logging.debug("Stopping triton event")
        pass

    def kernel_finished(self):
        logging.debug("Checking if kernel has finished")
        pass

    def run_kernel(self, func, gpu_args, threads, grid, stream):
        # Run the kernel
        logging.debug("Running triton kernel")
        pass


