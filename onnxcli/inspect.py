import logging
from onnxcli.common import SubCmd, dtype, shape

logger = logging.getLogger('onnxcli')


class InspectCmd(SubCmd):
    """Prints the information of nodes tensors of the given model"""

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
            '--indices',
            type=int,
            nargs="+",
            default=[],
            help="Specify the indices of the node(s) or tensor(s) to inspect." " Can NOT set together with --names",
        )
        subparser.add_argument(
            '-N',
            '--names',
            type=str,
            nargs="+",
            default=[],
            help="Specify the names of the node(s) or tensor(s) to inspect." " Can NOT set together with --indices",
        )
        subparser.add_argument(
            '-d',
            '--detail',
            action='store_true',
            help="Print detailed information of the nodes or tensors that specified by --indices or --names."
            " Warning: will print the data of tensors.",
        )

    def run(self, args):
        logger.info("Running <Inspect> on model {}".format(args.input_path))
        has_indices = len(args.indices) != 0
        has_names = len(args.names) != 0
        no_tensor_or_node = args.node is None and args.tensor is None
        if has_indices and has_names:
            raise ValueError("Can NOT set both --indices and --names")
        if (has_indices or has_indices) and no_tensor_or_node:
            raise ValueError("Can NOT set --indices or --names without --node or --tensor")
        if (not has_indices and not has_names) and args.detail:
            raise ValueError("Can NOT set --detail without --indices or --names")

        import onnx

        try:
            onnx.checker.check_model(args.input_path)
        except Exception:
            logger.warn("Failed to check model {}, statistic could be inaccurate!".format(args.input_path))
        m = onnx.load_model(args.input_path)
        g = m.graph

        self.print_basic(args, g)

        if args.meta:
            self.print_meta(m)

        if args.node:
            self.print_nodes(g, args.indices, args.names, args.detail)

        if args.tensor:
            self.print_tensor(g, args.indices, args.names, args.detail)

    def print_meta(self, m):
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

    def print_basic(self, args, g):
        print("Inpect of model {}".format(args.input_path))
        print("=" * 80)
        print("  Graph name: {}".format(len(g.name)))
        print("  Graph inputs: {}".format(len(g.input)))
        print("  Graph outputs: {}".format(len(g.output)))
        print("  Nodes in total: {}".format(len(g.node)))
        print("  ValueInfo in total: {}".format(len(g.value_info)))
        print("  Initializers in total: {}".format(len(g.initializer)))
        print("  Sparse Initializers in total: {}".format(len(g.sparse_initializer)))
        print("  Quantization in total: {}".format(len(g.quantization_annotation)))

    def print_tensor(self, g, indices, names, detail):
        print("\nTensor information:")
        print("-" * 80)

        def print_value_info(t):
            txt = "  ValueInfo \"{}\":".format(t.name)
            txt += " type {},".format(dtype(t.type.tensor_type.elem_type))
            txt += " shape {},".format(shape(t.type.tensor_type.shape))
            print(txt)

        def print_initializer(t, detail):
            txt = "  Initializer \"{}\":".format(t.name)
            txt += " type {},".format(dtype(t.data_type))
            txt += " shape {},".format(t.dims)
            print(txt)
            if detail:
                print("    float data: {}".format(t.float_data))

        # print with indices
        if len(indices) > 0:
            for idx in indices:
                if idx >= len(g.value_info) and idx >= len(g.initializer):
                    raise ValueError(
                        "indices {} out of range, value_info in total {}, initializer in total {}".format(
                            idx, len(g.value_info), len(g.initializer)
                        )
                    )
                print_value_info(g.value_info[idx])
                print_initializer(g.initializer[idx], detail)
            return

        # print with names
        if len(names) > 0:
            found_any = False
            for name in names:
                for i in g.value_info:
                    if i.name == name:
                        print_value_info(i)
                        found_any = True
                        break
                for i in g.initializer:
                    if i.name == name:
                        print_initializer(i, detail)
                        found_any = True
                        break
            if not found_any:
                raise ValueError("No tensor found with name {}".format(name))
            return

        # print all tensors
        for t in g.value_info:
            print_value_info(t)
        for t in g.initializer:
            print_initializer(t, False)

    def print_nodes(self, g, indices, names, detail):
        print("\nNode information:")
        print("-" * 80)

        def print_node(n, detail):
            txt = "  Node \"{}\":".format(n.name)
            txt += " type \"{}\",".format(n.op_type)
            txt += " inputs \"{}\",".format(n.input)
            txt += " outputs \"{}\"".format(n.output)
            print(txt)
            if detail:
                print("    attributes: {}".format(n.attribute))

        # print with indices
        if len(indices) > 0:
            for idx in indices:
                if idx >= len(g.node):
                    raise ValueError("indices {} out of range, node in total {}".format(idx, len(g.node)))
                print_node(g.node[idx], detail)
            return

        # print with names
        if len(names) > 0:
            found_any = False
            for name in names:
                for n in g.node:
                    if n.name == name:
                        print_node(n, detail)
                        found_any = True
                        break
            if not found_any:
                raise ValueError("No node found with name {}".format(name))
            return

        import collections

        ops = collections.Counter([node.op_type for node in g.node])
        for op, count in ops.most_common():
            print("  Node type \"{}\" has: {}".format(op, count))

        print("-" * 80)
        for node in g.node:
            print(print_node(node, False))
