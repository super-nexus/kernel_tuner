import inspect
import ast
import copy

from typing import Any
from kernel_tuner.kernel_sources.kernel_source import KernelSource
from kernel_tuner.kernel_sources.model.prepared_kernel_source_data import PreparedKernelSourceData


class KernelSourceFn(KernelSource):

    def __init__(self, kernel_name, kernel_sources, lang, defines=None):
        super().__init__(kernel_name, kernel_sources, lang, defines)
        self.source_kernel_fn = kernel_sources[0]
        self.kernel_fn = self.source_kernel_fn

    def prepare_kernel_instance(self, kernel_options, params, grid, threads):
        new_kernel_fn = self.apply_params_to_source_fn(params)
        self.kernel_fn = new_kernel_fn

        return PreparedKernelSourceData(
            temp_files=None,
            kernel_name=self.kernel_name,
            kernel_fn=new_kernel_fn,
            kernel_str=None
        )

    def check_argument_lists(self, kernel_name, arguments):
        return True

    def apply_params_to_source_fn(self, params):
        transformer = ReplaceVars(params)
        fn_copy = copy.deepcopy(self.source_kernel_fn)
        new_tree = transformer.visit(fn_copy)
        ast.fix_missing_locations(new_tree)

        new_code = compile(new_tree, filename="<ast>", mode="exec")

        new_namespace = {}
        exec(new_code, globals(), new_namespace)
        new_fn = new_namespace[self.kernel_name]

        return new_fn


class ReplaceVars(ast.NodeTransformer):

    def __init__(self, params: dict):
        self.params = params

    def visit_Name(self, node: ast.Name) -> Any:
        if isinstance(node.ctx, ast.Load) and node.id in self.params.keys():
            return ast.copy_location(
                ast.Constant(value=self.params[node.id]),
                node
            )

        return node
