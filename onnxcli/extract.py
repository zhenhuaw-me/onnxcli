import logging
import onnx
import os

from onnxcli.common import SubCmd

logger = logging.getLogger('onnxcli')


class ExtractCmd(SubCmd):
    """Extract sub model that is determined by given input and output tensor names.

    The sub-model is defined by the names of the input and output tensors *exactly*.
    Note: For control-flow operators, e.g. If and Loop, the _boundary of sub-model_,
    which is defined by the input and output tensors, should not _cut through_ the
    subgraph that is connected to the _main graph_ as attributes of these operators.
    """

    subcmd = 'extract'

    def add_args(self, subparser):
        subparser.add_argument('input_path', type=str, help="The path to original ONNX model")
        subparser.add_argument('output_path', type=str, default=None, help="The path to save the extracted ONNX model")
        subparser.add_argument(
            '-i',
            '--input_names',
            nargs='+',
            help="The names of the input tensors that to be extracted.",
        )
        subparser.add_argument(
            '-o',
            '--output_names',
            nargs='+',
            help="The names of the output tensors that to be extracted.",
        )

    def run(self, args):
        logger.info("Running <Model Extraction> on model {}".format(args.input_path))
        if not os.path.exists(args.input_path):
            raise ValueError("Invalid input model path: {}".format(args.input_path))
        if args.output_path is None:
            raise ValueError("Output model path shall not be empty!")
        if len(args.output_names) == 0:
            raise ValueError("Output tensor names shall not be empty!")
        try:
            onnx.checker.check_model(args.input_path)
        except Exception as e:
            logger.warning("Input model invalid, the resulted model can be invalid too!\n {}".format(e))

        model = onnx.load(args.input_path)
        e = Extractor(model)
        extracted = e.extract(args.input_names, args.output_names)
        onnx.save(extracted, args.output_path)


class Extractor:
    def __init__(self, model):
        self.model = onnx.shape_inference.infer_shapes(model)
        self.graph = self.model.graph
        self.wmap = self._build_name2obj_dict(self.graph.initializer)
        self.vimap = self._build_name2obj_dict(self.graph.value_info)

    @staticmethod
    def _build_name2obj_dict(objs):
        return {obj.name: obj for obj in objs}

    def _collect_new_io_core(self, original_io, io_names_to_extract):
        original_io_map = self._build_name2obj_dict(original_io)
        original_io_names = set(original_io_map.keys())
        s_io_names_to_extract = set(io_names_to_extract)
        io_names_to_keep = s_io_names_to_extract & original_io_names
        new_io_names_to_add = s_io_names_to_extract - original_io_names

        new_io_tensors = []
        for name in io_names_to_keep:
            new_io_tensors.append(original_io_map[name])
        for name in new_io_names_to_add:
            # activation become input or output
            new_io_tensors.append(self.vimap[name])

        # adjust sequence
        new_io_tensors_map = self._build_name2obj_dict(new_io_tensors)
        return [new_io_tensors_map[name] for name in io_names_to_extract]

    def _collect_new_inputs(self, names):
        return self._collect_new_io_core(self.graph.input, names)

    def _collect_new_outputs(self, names):
        return self._collect_new_io_core(self.graph.output, names)

    def _dfs_search_reachable_nodes(self, node_output_name, graph_input_names, reachable_nodes):
        if node_output_name in graph_input_names:
            return
        for node in self.graph.node:
            if node in reachable_nodes:
                continue
            if node_output_name not in node.output:
                continue
            reachable_nodes.append(node)
            for name in node.input:
                self._dfs_search_reachable_nodes(name, graph_input_names, reachable_nodes)

    def _collect_reachable_nodes(self, input_names, output_names):
        reachable_nodes = list()
        for name in output_names:
            self._dfs_search_reachable_nodes(name, input_names, reachable_nodes)
        # needs to be topology sorted.
        nodes = [n for n in self.graph.node if n in reachable_nodes]
        return nodes

    def _collect_reachable_tensors(self, nodes):
        all_tensors_name = set()
        for node in nodes:
            for name in node.input:
                all_tensors_name.add(name)
            for name in node.output:
                all_tensors_name.add(name)

        initializer = [self.wmap[t] for t in self.wmap.keys() if t in all_tensors_name]
        value_info = [self.vimap[t] for t in self.vimap.keys() if t in all_tensors_name]
        assert len(self.graph.sparse_initializer) == 0
        assert len(self.graph.quantization_annotation) == 0
        return (initializer, value_info)

    def _make_model(self, nodes, inputs, outputs, initializer, value_info):
        name = 'Extracted from {' + self.graph.name + '}'
        graph = onnx.helper.make_graph(nodes, name, inputs, outputs, initializer=initializer, value_info=value_info)

        meta = {
            'ir_version': self.model.ir_version,
            'opset_imports': self.model.opset_import,
            'producer_name': 'onnx.utils.extract_model',
        }
        return onnx.helper.make_model(graph, **meta)

    def extract(self, input_names, output_names):
        inputs = self._collect_new_inputs(input_names)
        outputs = self._collect_new_outputs(output_names)
        nodes = self._collect_reachable_nodes(input_names, output_names)
        initializer, value_info = self._collect_reachable_tensors(nodes)
        model = self._make_model(nodes, inputs, outputs, initializer, value_info)
        return model
