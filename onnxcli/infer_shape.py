import logging
from onnxcli.common import SubCmd

logger = logging.getLogger('onnxcli')


class InferShapeCmd(SubCmd):
    """Run Shape Inference on given ONNX model"""

    subcmd = 'infershape'

    def add_args(self, subparser):
        subparser.add_argument('input_path', type=str, help="The input ONNX model")
        subparser.add_argument('-o', '--output_path', required=False, type=str, help="The output ONNX model")

    def run(self, args):
        logger.info("Running <Shape Inference> on model {}".format(args.input_path))
        import onnx

        if args.output_path:
            onnx.shape_inference.infer_shapes_path(args.input_path, args.output_path)
        else:
            onnx.shape_inference.infer_shapes_path(args.input_path)
