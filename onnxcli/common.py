class SubCmd:
    subcmd = None

    def __init__(self, subparsers):
        if not self.subcmd:
            raise RuntimeError("subcmd must be valid")

        subparser = subparsers.add_parser(self.subcmd, help=self.__doc__)
        self.add_args(subparser)
        subparser.set_defaults(func=self.run)

    def add_args(self, subparser):
        """Add dedicated arguments to the subcommand.

        Needs to be overrided, and only be called by `__init__()`.
        """
        raise RuntimeError("{}.add_args() need to be overrided!".format(self.__name__))

    def run(self, args):
        """Run the utility with args.

        Won't be called explicitly.
        """
        raise RuntimeError("{}.run() need to be overrided!".format(self.__name__))


def dtype(Key):
    # sync with TensorProto.DataType of https:#github.com/onnx/onnx/blob/master/onnx/onnx.proto
    RawMap = [
        # fmt off
        'UNDEFINED',
        'FLOAT',
        'UINT8',
        'INT8',
        'UINT16',
        'INT16',
        'INT32',
        'INT64',
        'STRING',
        'BOOL',
        'FLOAT16',
        'DOUBLE',
        'UINT32',
        'UINT64',
        'COMPLEX64',
        'COMPLEX128',
        'BFLOAT16',
        # fmt on
    ]
    return RawMap[Key]


def shape(ShapeProto):
    return [d.dim_value for d in ShapeProto.dim]
