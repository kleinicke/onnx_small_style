model="model/candy1.model"
size="135"
image="birds"
type="gogh"

#python neural_style/neural_style.py eval --content-image images/content-images/${image}${size}.jpeg  --model ${model} --output-image results/${type}${size}.jpg --cuda 0 --export_onnx model/${type}${size}.onnx
# python neural_style/neural_style.py eval --content-image images/content-images/birds135.jpeg  --model "model/rain_princess.model" --output-image results/rain135.jpg --cuda 0 --export_onnx model/rain135.onnx
# python neural_style/neural_style.py eval --content-image images/content-images/birds350.jpeg  --model model/gogh_starry.model --output-image results/gogh_350.jpg --cuda 0 --export_coreml model/gogh.mlmodel
python neural_style/neural_style.py eval --content-image images/content-images/birds300.jpeg  --model "model/candytest.model" --output-image results/candy301.jpg --cuda 0 --export_onnx model/candy135.onnx