"""The main command dispatcher"""

import argparse
import logging
import sys

from onnxcli import __doc__ as DESCRIPTION
from onnxcli.check import CheckCmd
from onnxcli.convert import ConvertCmd
from onnxcli.draw import DrawCmd
from onnxcli.extract import ExtractCmd
from onnxcli.infer_shape import InferShapeCmd
from onnxcli.inspect import InspectCmd
from onnxcli.optimize import OptimizeCmd
from onnxcli.setup import SetupCmd

logger = logging.getLogger('onnxcli')


def dispatch():
    dispatch_core(sys.argv[1:])


def dispatch_core(*raw_args):
    logger.debug("Running {}".format(*raw_args))

    parser = argparse.ArgumentParser(description=DESCRIPTION)
    subparsers = parser.add_subparsers(title='subcommands')

    CheckCmd(subparsers)
    ConvertCmd(subparsers)
    DrawCmd(subparsers)
    ExtractCmd(subparsers)
    InferShapeCmd(subparsers)
    InspectCmd(subparsers)
    OptimizeCmd(subparsers)
    SetupCmd(subparsers)

    args = parser.parse_args(*raw_args)
    args.func(args)
