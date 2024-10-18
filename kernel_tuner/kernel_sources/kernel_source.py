import inspect
import kernel_tuner.util as util

from abc import abstractmethod

from kernel_tuner.kernel_sources.model.prepared_kernel_source_data import PreparedKernelSourceData


class KernelSource:

    def __init__(self, kernel_name, kernel_sources, lang, defines=None):
        if not isinstance(kernel_sources, list):
            kernel_sources = [kernel_sources]

        self.kernel_sources = kernel_sources
        self.kernel_name = kernel_name
        self.defines = defines
        self.lang = lang

    @abstractmethod
    def prepare_kernel_instance(self, kernel_options, params, grid, threads) -> PreparedKernelSourceData:
        raise NotImplementedError("create_kernel_instance not implemented")

    @abstractmethod
    def check_argument_lists(self, kernel_name, arguments):
        raise NotImplementedError("check_argument_lists not implemented")
