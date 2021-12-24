"""The main command dispatcher"""

import argparse
import sys
import logging

from onnxcli import __doc__ as DESCRIPTION
from onnxcli.infer_shape import InferShapeCmd
from onnxcli.extract import ExtractCmd
from onnxcli.inspect import InspectCmd
from onnxcli.draw import DrawCmd


logger = logging.getLogger('onnxcli')


def dispatch():
    dispatch_core(sys.argv[1:])


def dispatch_core(*raw_args):
    logger.debug("Running {}".format(*raw_args))

    parser = argparse.ArgumentParser(description=DESCRIPTION)
    subparsers = parser.add_subparsers(title='subcommands')

    # collect commands
    InferShapeCmd(subparsers)
    ExtractCmd(subparsers)
    InspectCmd(subparsers)
    DrawCmd(subparsers)

    args = parser.parse_args(*raw_args)
    args.func(args)
