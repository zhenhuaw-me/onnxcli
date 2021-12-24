import logging
import tempfile
import subprocess
import shlex
from onnxcli.common import SubCmd, dtype, shape

logger = logging.getLogger('onnxcli')


class DrawCmd(SubCmd):
    """Draw the graph with given ONNX model. Save you from Netron."""

    subcmd = 'draw'

    def add_args(self, subparser):
        subparser.add_argument('input_path', type=str, help="The input ONNX model path")
        subparser.add_argument('output_path', type=str, help="The output drawing graph path")
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
        import onnx

        m = onnx.load_model(input_path)
        dot_str = "digraph onnxcli {\n"

        # handle chars that are key in graphviz
        def fixname(name):
            return name.replace('\\', '\\\\').replace(':', '\\:')

        node_names = set()
        for node in m.graph.node:
            nname = fixname(node.name)
            node_names.add(nname)
            dot_str += '"{}" [label="{}\\n<{}>" fonstsize=16 shape=oval];\n'.format(nname, nname, node.op_type)
            for iname in node.input:
                dot_str += '  "{}" -> "{}";\n'.format(fixname(iname), nname)

        # tensors
        for tensor in m.graph.initializer:
            tname = fixname(tensor.name)
            dot_str += '"{}" [label="{}\\n{}, {}" fonstsize=10 style=rounded shape=rectangle];\n'.format(
                tname, tname, dtype(tensor.data_type), tensor.dims
            )
        for tensor in m.graph.value_info:
            tname = fixname(tensor.name)
            if tname not in node_names:
                dot_str += '"{}" [label="{}\\n{}, {}" fonstsize=10 shape=rectangle];\n'.format(
                    tname, tname, dtype(tensor.type.tensor_type.elem_type), shape(tensor.type.tensor_type.shape)
                )

        dot_str += "}\n"
        return dot_str
