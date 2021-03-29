model="model/gogh_1_Fri_Feb_26_13_59_20_2021.model"
size="135"
image="birds"
type="gogh"

#python neural_style/neural_style.py eval --content-image images/content-images/${image}${size}.jpeg  --model ${model} --output-image results/${type}${size}.jpg --cuda 0 --export_onnx model/${type}${size}.onnx
python neural_style/neural_style.py eval --content-image images/content-images/birds135.jpeg  --model "model/rain_princess.model" --output-image results/rain135.jpg --cuda 0 --export_onnx model/rain135.onnx