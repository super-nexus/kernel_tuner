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

        self.device_id = torch.cuda.current_device()

        self.device_properties = torch.cuda.get_device_properties(self.device_id)
        self.name = torch.cuda.get_device_name(self.device_id)
        self.max_threads = self.device_properties.max_threads_per_multi_processor

        self.stream = torch.cuda.default_stream()
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.stop_event = torch.cuda.Event(enable_timing=True)

        super().__init__(device=device, iterations=iterations, compiler_options=compiler_options, observers=observers)

    def ready_argument_list(self, arguments):
        # Allocate memory here
        torch_args = []

        for arg in arguments:
            if isinstance(arg, torch.Tensor):
                torch_args.append(arg.cuda())
            elif isinstance(arg, np.ndarray):
                torch_arg = torch.from_numpy(arg)
                torch_arg_gpu = torch_arg.cuda()
                torch_args.append(torch_arg_gpu)
            elif isinstance(arg, np.generic):
                scalar_value = arg.item()
                torch_args.append(scalar_value)
            else:
                logger.warning("Unknown instance in triton functions")

        return torch_args

    def compile(self, kernel_instance):
        logging.debug("Compiling triton kernel")
        if kernel_instance.kernel_source.is_callable:
            func = kernel_instance.kernel_source.kernel_sources[0]
            return triton.jit(func)
        else:
            raise NotImplmenentedError("Currently Triton only supports passing down a callable function")

    def start_event(self):
        logging.debug("Start triton event")
        self.start_event.record()

    def stop_event(self):
        logging.debug("Stop triton event")
        self.stop_event.record()

    def kernel_finished(self):
        logging.debug("Checking if kernel has finished")
        return self.stop_event.query()

    def run_kernel(self, func, gpu_args, threads, grid, stream=None):
        # Run the kernel
        if stream is None:
            stream = self.stream

        with torch.cuda.stream(stream):
            logging.debug("Running triton kernel")
            func[grid](*gpu_args, BLOCK_SIZE=threads[0])

    def synchronize(self):
        torch.cuda.synchronize()

    def memset(self, allocation, value, size):
        pass

    def memcpy_dtoh(self, dest, src):
        pass

    def memcpy_htod(self, dest, src):
        pass

    def copy_constant_memory_args(self, cmem_args):
        raise NotImplementedError("Triton does not support constant memory")

    def copy_shared_memory_args(self, smem_args):
        raise NotImplementedError("Triton does not support shared memory")

    def copy_texture_memory_args(self, texmem_args):
        raise NotImplementedError("Triton does not support texture memory")

    units = {"time": "ms", "power": "s,mW", "energy": "J"}
