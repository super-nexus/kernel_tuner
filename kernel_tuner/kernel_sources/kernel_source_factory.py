import inspect

from kernel_tuner.kernel_sources.kernel_source_fn import KernelSourceFn
from kernel_tuner.kernel_sources.kernel_source_str import KernelSourceStr
from kernel_tuner.language import Language


def get_kernel_source(kernel_name, kernel_source, lang, defines):
    if inspect.isfunction(kernel_source) and lang.upper() == Language.TRITON:
        return KernelSourceFn(kernel_name, kernel_source, lang, defines)
    else:
        return KernelSourceStr(kernel_name, kernel_source, lang, defines)
