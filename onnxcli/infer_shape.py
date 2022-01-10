import logging
import onnx
from onnxcli.common import SubCmd

logger = logging.getLogger('onnxcli')


class InferShapeCmd(SubCmd):
    """Run Shape Inference on given ONNX model."""

    subcmd = 'infershape'

    def add_args(self, subparser):
        subparser.add_argument('input_path', type=str, help="The path to the input ONNX model")
        subparser.add_argument('output_path', type=str, help="The path to the output ONNX model")

    def run(self, args):
        logger.info("Running <Shape Inference> on model {}".format(args.input_path))

        if args.output_path:
            onnx.shape_inference.infer_shapes_path(args.input_path, args.output_path)
        else:
            onnx.shape_inference.infer_shapes_path(args.input_path)
