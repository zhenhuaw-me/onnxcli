import logging
import tempfile
import subprocess
import shlex
import onnx
from onnxcli.common import SubCmd, dtype, shape

logger = logging.getLogger('onnxcli')


class DrawCmd(SubCmd):
    """Draw the graph topology in [svg, dot, png] formats of the given ONNX model.

    Give you quick view of the attributes of the tensors and nodes in addition.
    In the figure the node is ellipse and tensor is rectangle (the rounded ones are initializers).
    The generated figures can be viewed in browser or image viewer without waiting for the model to load.
    It's really hepful for when investigating large models - save you from Netron.
    """

    subcmd = 'draw'

    def add_args(self, subparser):
        subparser.add_argument('input_path', type=str, help="The path to the input ONNX model")
        subparser.add_argument('output_path', type=str, help="The path to the output drawing graph")
        subparser.add_argument(
            '-t',
            '--type',
            type=str,
            default='svg',
            choices=['svg', 'dot', 'png', 'jpg', 'ps'],
            help="The type of the drawing. Default is svg. Require dot (graphviz) to be installed.",
        )

    def run(self, args):
        logger.info("Running <Graph Drawing> on model {}".format(args.input_path))

        dot_str = self.gen_graphviz_str(args.input_path)
        if args.type == 'dot':
            with open(args.output_path, 'w') as f:
                f.write(dot_str)
        else:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.dot') as tmpfp:
                tmpfp.write(dot_str)
                tmpfp.flush()
                tmpfp.seek(0)
                dot_cmd = 'dot -T{} {} -o {}'.format(args.type, tmpfp.name, args.output_path)
                logger.debug("Invoking CMD {}".format(dot_cmd))
                try:
                    subprocess.call(shlex.split(dot_cmd))
                except OSError:
                    raise OSError("dot is not installed. Please install graphviz and try again.")
        logger.info("Drawing graph is saved to {}".format(args.output_path))

    def gen_graphviz_str(self, input_path):
        logger.debug("Generating graphviz string from {}".format(input_path))

        # handle chars that are key in graphviz
        def fixname(name):
            return name.replace('\\', '\\\\').replace(':', '\\:')

        # sometimes, a tensor the same name as a node, which will cause issue when rendering the graph
        # key used as graphviz node key to build graph, name used as graphviz node label
        def tensor_key(name):
            return 'tensor_' + fixname(name)

        def node_key(name):
            return 'node_' + fixname(name)

        m = onnx.load_model(input_path)
        dot_str = "digraph onnxcli {\n"

        # Keep track of the original tensor names.
        # Many tensors are not recorded in graph.value_info, thus we need to generate edge for them specially.
        tensor_names = set()

        # nodes
        for node in m.graph.node:
            nname = fixname(node.name)
            nkey = node_key(node.name)
            dot_str += '"{}" [label="{}\\n<{}>" fonstsize=16 shape=oval];\n'.format(nkey, nname, node.op_type)
            for iname in node.input:
                tensor_names.add(iname)
                dot_str += '  "{}" -> "{}";\n'.format(tensor_key(iname), nkey)
            for oname in node.output:
                tensor_names.add(oname)
                dot_str += '  "{}" -> "{}";\n'.format(nkey, tensor_key(oname))

        # tensors
        for tensor in m.graph.initializer:
            tensor_names.remove(tensor.name)
            dot_str += '"{}" [label="{}\\n{}, {}" fonstsize=10 style=rounded shape=rectangle];\n'.format(
                tensor_key(tensor.name), fixname(tensor.name), dtype(tensor.data_type), tensor.dims
            )
        all_value_info = list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output)
        for tensor in all_value_info:
            if tensor.name in tensor_names:
                tensor_names.remove(tensor.name)
                dot_str += '"{}" [label="{}\\n{}, {}" fonstsize=10 shape=rectangle];\n'.format(
                    tensor_key(tensor.name),
                    fixname(tensor.name),
                    dtype(tensor.type.tensor_type.elem_type),
                    shape(tensor.type.tensor_type.shape),
                )

        if len(tensor_names) != 0:
            # the tensors that are not in graph.initializer nor graph.value_info
            logger.warning("There are tensors that only have name (no data type or shape) in the graph.")
            for n in tensor_names:
                dot_str += '"{}" [label="{}" fonstsize=10 shape=rectangle];\n'.format(tensor_key(n), fixname(n))

        dot_str += "}\n"
        return dot_str
