from onnxcli.common import SubCmd


class InferShapeCmd(SubCmd):
    """Run Shape Inference on given ONNX model"""
    subcmd = 'infershape'

    def add_args(self, subparser):
        subparser.add_argument('input_model', type=str, help="The input ONNX model")
        subparser.add_argument('-o', required=False, type=str, help="The output ONNX model")

    def run(self, args):
        print("Running <Shape Inference> on model {}".format(args.input_model))
        import onnx
        if args.o:
            onnx.shape_inference.infer_shapes_path(args.input_model, args.o)
        else:
            onnx.shape_inference.infer_shapes_path(args.input_model)
