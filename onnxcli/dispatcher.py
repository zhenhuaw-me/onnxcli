"""The main command dispatcher"""

import argparse
from onnxcli import DESCRIPTION


def infer_shape(args):
    print("Running <Shape Inference> on model {}".format(args.input_model))
    import onnx
    if args.o:
        onnx.shape_inference.infer_shapes_path(args.input_model, args.o)
    else:
        onnx.shape_inference.infer_shapes_path(args.input_model)


def infer_shape_arg(subparsers):
    shape_parser = subparsers.add_parser('infershape', help="Run Shape Inference on given ONNX model")
    shape_parser.add_argument('input_model', type=str, help="The input ONNX model")
    shape_parser.add_argument('-o', required=False, type=str, help="The output ONNX model")
    shape_parser.set_defaults(func=infer_shape)


def dispatch():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    subparsers = parser.add_subparsers(title='subcommands')

    infer_shape_arg(subparsers)

    args = parser.parse_args()
    args.func(args)
