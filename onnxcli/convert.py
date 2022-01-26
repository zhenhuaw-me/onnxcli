import logging
import onnx
import os
from onnxcli.common import SubCmd

logger = logging.getLogger('onnxcli')


class ConvertCmd(SubCmd):
    """Convert the given model to or from ONNX."""

    subcmd = 'convert'

    def add_args(self, subparser):
        subparser.add_argument('input_path', type=str, help="The path to the input model")
        subparser.add_argument('output_path', type=str, help="The path to the output model")
        subparser.add_argument(
            '-i', '--input_type', type=str, default='onnx', choices=['onnx'], help="The type of the input model"
        )
        subparser.add_argument(
            '-o', '--output_type', type=str, default='onnx', choices=['json'], help="The type of the output model"
        )

    def run(self, args):
        logger.info("Running <Converter> on model {}".format(args.input_path))

        if not os.path.exists(args.input_path):
            raise ValueError("Input model file not existed: {}".format(args.input_path))
        if args.input_type == 'onnx' and args.output_type == 'json':
            onnx2json(args.input_path, args.output_path)
        else:
            raise NotImplementedError(
                "Conversion from {} to {} is not supported yet.".format(args.input_type, args.output_type)
            )


def onnx2json(input_path, output_path):
    """Convert the given ONNX model to JSON."""
    logger.info("Converting <ONNX to JSON> on model {}".format(input_path))
    import json

    try:
        from google.protobuf.json_format import MessageToJson
    except ImportError as err:
        logger.error("Failed to import protobuf. Try to fix with `onnx setup`.")
        raise err

    m = onnx.load(input_path)
    msg = MessageToJson(m)
    j = json.loads(msg)
    with open(output_path, 'w') as f:
        json.dump(j, f, indent=4)
    logger.info("JSON model saved as {}".format(output_path))
