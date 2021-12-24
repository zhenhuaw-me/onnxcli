import logging
import tempfile
import subprocess
import shlex
import re
from onnxcli.common import SubCmd

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
        import onnx

        def gen_graphviz(input_path):
            m = onnx.load_model(input_path)
            dot_str = "digraph graphname {\n"
            for node in m.graph.node:
                output_name = node.name
                color = ""
                if node.op_type.startswith("_"):
                    color = ' color="yellow"'
                if node.op_type == "CELL":
                    color = ' color="red"'
                dot_str += '"{}" [label="{},{}"{}];\n'.format(output_name, node.op_type, node.name, color)
                for input_name in node.input:
                    parts = input_name.split(":")
                    input_name = re.sub(r"^\^", "", parts[0])
                    dot_str += '  "{}" -> "{}";\n'.format(input_name, output_name)
            dot_str += "}\n"
            return dot_str

        dot_str = gen_graphviz(args.input_path)
        if args.type == 'dot':
            with open(args.output_path, 'w') as f:
                f.write(dot_str)
        else:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.dot') as tmpfp:
                tmpfp.write(dot_str)
                tmpfp.flush()
                tmpfp.seek(0)
                dot_cmd = 'dot -T{} {} -o {}'.format(args.type, tmpfp.name, args.output_path)
                logger.debug("Running {}".format(dot_cmd))
                try:
                    subprocess.call(shlex.split(dot_cmd))
                except OSError:
                    raise OSError("dot is not installed. Please install graphviz and try again.")
        logging.info("Drawing graph is saved to {}".format(args.output_path))
