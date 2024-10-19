import inspect
import ast
import copy
import uuid

import astor
import tempfile
import importlib.util

from typing import Any

from kernel_tuner.kernel_sources.kernel_source import KernelSource
from kernel_tuner.kernel_sources.model.prepared_kernel_source_data import PreparedKernelSourceData


class KernelSourceFn(KernelSource):

    def __init__(self, kernel_name, kernel_source, lang, defines=None):
        super().__init__(kernel_name, kernel_source, lang, defines)
        if isinstance(kernel_source, list):
            raise ValueError("KernelSourceFn only supports a single kernel source function")

        self.source_kernel_fn = kernel_source
        self.kernel_fn = self.source_kernel_fn
        self.source = inspect.getsource(kernel_source)
        self.source_tree = ast.parse(self.source)
        self.triton_import_node = ast.ImportFrom(
            module='triton',
            names=[ast.alias(name='language', asname='tl')],
            level=0
        )

    def prepare_kernel_instance(self, kernel_options, params, grid, threads):
        new_kernel_fn, temp_file_path = self.apply_params_to_source_fn(params)
        self.kernel_fn = new_kernel_fn

        return PreparedKernelSourceData(
            temp_files=[temp_file_path],
            kernel_name=self.kernel_name,
            kernel_fn=new_kernel_fn,
            kernel_str=None
        )

    def check_argument_lists(self, kernel_name, arguments):
        return True

    def apply_params_to_source_fn(self, params):
        transformer = ReplaceVars(params)
        source_tree_copy = copy.deepcopy(self.source_tree)
        new_tree = transformer.visit(source_tree_copy)
        new_tree.body.insert(0, self.triton_import_node)

        ast.fix_missing_locations(new_tree)
        new_source = astor.to_source(new_tree)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(new_source)
            temp_file_path = temp_file.name

        module_name = f'temp_kernel_module_{uuid.uuid4().hex}'
        spec = importlib.util.spec_from_file_location(module_name, temp_file_path)
        temp_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(temp_module)
        new_fn = getattr(temp_module, self.kernel_name)

        return new_fn, temp_file_path


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
