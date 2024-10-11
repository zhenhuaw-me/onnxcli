import logging
import onnx
from onnxcli.common import SubCmd

logger = logging.getLogger('onnxcli')


class CompareCmd(SubCmd):
    """Compare if the 2 given ONNX models have the same topology."""

    subcmd = 'compare'

    def add_args(self, subparser):
        subparser.add_argument('path1', type=str, help="The path to the first ONNX model.")
        subparser.add_argument('path2', type=str, help="The path to the second ONNX model.")

    def run(self, args):
        logger.info(f"Running <Compare> on model {args.path1} {args.path2}")

        model1 = onnx.load(args.path1)
        model2 = onnx.load(args.path2)
        compare_topology(model1.graph, model2.graph)


def compare_topology(g1, g2):
    """Compare the topology of 2 ONNX models."""
    if len(g1.node) != len(g2.node):
        logger.info("The 2 models have different number of nodes.")
        return False

    for node1, node2 in zip(g1.node, g2.node):
        logger.debug(f"Comparing node {node1.name} vs {node2.name}")
        if node1.op_type != node2.op_type:
            logger.info(f"The 2 nodes have different op_type: {node1.op_type} vs {node2.op_type}")
            return False
        if len(node1.input) != len(node2.input) or len(node1.output) != len(node2.output):
            logger.info("The 2 nodes have different number of inputs or outputs.")
            return False

    # TBD
    return True


def compare_inputs(g1, g2):
    inputs1 = g1.input
    inputs2 = g2.input

    if len(inputs1) != len(inputs2):
        logger.info("The 2 models have different number of inputs.")
        return False

    for input1, input2 in zip(inputs1, inputs2):
        dtype1 = input1.type.tensor_type.elem_type
        dtype2 = input2.type.tensor_type.elem_type
        if dtype1 != dtype2:
            logger.info(f"The 2 inputs have different data types: {dtype1} vs {dtype2}")
            return False

        shape1 = [dim.dim_value for dim in input1.type.tensor_type.shape.dim]
        shape2 = [dim.dim_value for dim in input2.type.tensor_type.shape.dim]
        if shape1[:] != shape2[:]:
            logger.info(f"The 2 inputs have different shapes: {shape1} vs {shape2}")
            return False
    return True
