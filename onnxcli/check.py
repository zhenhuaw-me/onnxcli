import logging
import onnx
from onnxcli.common import SubCmd

logger = logging.getLogger('onnxcli')


class CheckCmd(SubCmd):
    """Check if the given ONNX model is semantically correct."""

    subcmd = 'check'

    def add_args(self, subparser):
        subparser.add_argument('path', type=str, help="The path to the ONNX model")

    def run(self, args):
        logger.info("Running <Checker> on model {}".format(args.path))

        onnx.checker.check_model(args.path)
