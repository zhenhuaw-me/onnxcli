import logging
import os
from onnxcli.common import SubCmd

logger = logging.getLogger('onnxcli')


class OptimizeCmd(SubCmd):
    """Optimize given ONNX model."""

    subcmd = 'optimize'

    def add_args(self, subparser):
        subparser.add_argument('input_path', type=str, help="The input ONNX model")
        subparser.add_argument('output_path', type=str, help="The output ONNX model")
        subparser.add_argument(
            '-p',
            '--passes',
            type=str,
            nargs="+",
            default=[],
            help="The passes to run with the optimizer. Will tip avaiable passes if not specified.",
        )

    def run(self, args):
        logger.info("Running <Optimization> on model {}".format(args.input_path))
        import onnx
        import onnxoptimizer

        if not os.path.exists(args.input_path):
            raise ValueError("Invalid input model path: {}".format(args.input_path))
        if len(args.passes) == 0:
            passes = onnxoptimizer.get_available_passes()
            logger.warn("No optimization passes specified, running all available passes: {}".format(passes))
        else:
            passes = args.passes
            logger.info("Running with passes: {}".format(passes))

        model = onnx.load(args.input_path)
        optimized = onnxoptimizer.optimize(model, passes)
        onnx.save(optimized, args.output_path)
