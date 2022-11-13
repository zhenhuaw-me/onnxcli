import logging
import subprocess
from onnxcli.common import SubCmd

logger = logging.getLogger('onnxcli')


class SetupCmd(SubCmd):
    """Setup python package."""

    subcmd = 'setup'

    def add_args(self, subparser):
        subparser.add_argument('-l', '--list', action='store_true', help="List the packages only.")

    def run(self, args):
        logger.info("Running <Package Setup>")

        pkgs = ['onnxoptimizer==0.2.7', 'protobuf']

        if args.list:
            print("Dependent packages:")
            for pkg in pkgs:
                print(pkg)
            return

        for pkg in pkgs:
            logger.info("Installing {}".format(pkg))
            subprocess.check_call(["pip", "install", pkg])
