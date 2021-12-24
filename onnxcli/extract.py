import logging
from onnxcli.common import SubCmd

logger = logging.getLogger('onnxcli')


class ExtractCmd(SubCmd):
    """Extract sub model that is determined by given input and output tensor names."""

    subcmd = 'extract'

    def add_args(self, subparser):
        subparser.add_argument('input_path', type=str, help="The path to original ONNX model")
        subparser.add_argument('output_path', type=str, help="The path to save the extracted ONNX model")
        subparser.add_argument(
            '-i',
            '--input_names',
            nargs="+",
            help="The names of the input tensors that to be extracted.",
        )
        subparser.add_argument(
            '-o',
            '--output_names',
            nargs="+",
            help="The names of the output tensors that to be extracted.",
        )

    def run(self, args):
        logger.info("Running <Model Extraction> on model {}".format(args.input_path))
        import onnx

        onnx.utils.extract_model(args.input_path, args.output_path, args.input_names, args.output_names)
