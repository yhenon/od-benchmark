"""Create a deterministic class-zero ONNX model for evaluator smoke tests."""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 32, 32])
output_info = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 10])
weights = numpy_helper.from_array(
    np.zeros((3 * 32 * 32, 10), dtype=np.float32), name="weights"
)
nodes = [
    helper.make_node("Flatten", ["input"], ["flat_input"], axis=1),
    helper.make_node("MatMul", ["flat_input", "weights"], ["logits"]),
]
graph = helper.make_graph(
    nodes,
    "constant-class-zero",
    [input_info],
    [output_info],
    initializer=[weights],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
onnx.checker.check_model(model)
onnx.save(model, Path(__file__).with_name("model.onnx"))
