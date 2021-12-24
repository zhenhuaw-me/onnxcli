import logging
from onnxcli.common import SubCmd, dtype, shape

logger = logging.getLogger('onnxcli')


class InspectCmd(SubCmd):
    """Prints the statistic of nodes tensors of the given model"""

    subcmd = 'inspect'

    def add_args(self, subparser):
        subparser.add_argument('input_path', type=str, help="The input ONNX model")
        subparser.add_argument(
            '-m',
            '--meta',
            action='store_true',
            help="Print the meta information of the model",
        )
        subparser.add_argument(
            '-n',
            '--node',
            action='store_true',
            help="Print the node information of the model",
        )
        subparser.add_argument(
            '-t',
            '--tensor',
            action='store_true',
            help="Print the tensor information of the model",
        )
        subparser.add_argument(
            '-i',
            '--index',
            type=int,
            default=None,
            help="Specify the index of the node or tensor to inspect",
        )

    def run(self, args):
        logger.info("Running <Inspect> on model {}".format(args.input_path))
        import onnx

        try:
            onnx.checker.check_model(args.input_path)
        except Exception:
            logger.warn(
                "Failed to check model {}, statistic could be inaccurate!".format(args.input_path)
            )

        print("Inpect of model {}".format(args.input_path))
        print("=" * 80)
        m = onnx.load_model(args.input_path)
        g = m.graph

        print("  Graph name: {}".format(len(g.name)))
        print("  Graph inputs: {}".format(len(g.input)))
        print("  Graph outputs: {}".format(len(g.output)))
        print("  Nodes in total: {}".format(len(g.node)))
        print("  ValueInfo in total: {}".format(len(g.value_info)))
        print("  Initializers in total: {}".format(len(g.initializer)))
        print("  Sparse Initializers in total: {}".format(len(g.sparse_initializer)))
        print("  Quantization in total: {}".format(len(g.quantization_annotation)))

        if args.meta:
            print("\nMeta information:")
            print("-" * 80)
            print("  IR Version: {}".format(m.ir_version))
            print("  Opset Import: {}".format(m.opset_import))
            print("  Producer name: {}".format(m.producer_name))
            print("  Producer version: {}".format(m.producer_version))
            print("  Domain: {}".format(m.domain))
            print("  Doc string: {}".format(m.doc_string))
            for i in m.metadata_props:
                print("  meta.{} = {}", i.key, i.value)

        if args.node:
            self.print_node(g, args.index)

        if args.tensor:
            self.print_tensor(g, args.index)

    def print_tensor(self, g, idx=None):
        print("\nTensor information:")
        print("-" * 80)

        def str_value_info(vi):
            txt = "  ValueInfo \"{}\":".format(vi.name)
            txt += " type {},".format(dtype(v.type.tensor_type.elem_type))
            txt += " shape {},".format(shape(v.type.tensor_type.shape))
            return txt

        def str_initializer(i):
            txt = "  Initializer \"{}\":".format(i.name)
            txt += " type {},".format(dtype(i.data_type))
            txt += " shape {},".format(i.dims)
            return txt

        if idx is not None:
            if idx >= len(g.value_info) and idx >= len(g.initializer):
                raise ValueError(
                    "Index {} out of range, value_info in total {}, initializer in total {}".format(
                        idx, len(g.value_info), len(g.initializer)
                    )
                )
            print(str_value_info(g.value_info[idx]))
            print(str_initializer(g.initializer[idx]))
            return

        for v in g.value_info:
            print(str_value_info(v))
        for i in g.initializer:
            print(str_initializer(i))

    def print_node(self, g, idx=None):
        print("\nNode information:")
        print("-" * 80)

        def str_node(n):
            txt = "  Node \"{}\":".format(n.name)
            txt += " type \"{}\",".format(n.op_type)
            txt += " inputs \"{}\",".format(n.input)
            txt += " outputs \"{}\"".format(n.output)
            return txt

        if idx is not None:
            if idx >= len(g.node):
                raise ValueError(
                    "Index {} out of range, nodes in total {}".format(idx, len(g.node))
                )
            print(str_node(g.node[idx]))
            return

        import collections

        ops = collections.Counter([node.op_type for node in g.node])
        for op, count in ops.most_common():
            print("  Node type \"{}\" has: {}".format(op, count))

        print("-" * 80)

        for node in g.node:
            print(str_node(node))
