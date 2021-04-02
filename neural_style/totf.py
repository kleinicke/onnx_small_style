import onnx
from onnx_tf.backend import prepare

onnx_model = onnx.load("input_path")  # load onnx model
output = prepare(onnx_model).run(input)  # run the loaded model