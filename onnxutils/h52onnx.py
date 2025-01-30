import tensorflow as tf
import tf2onnx
import onnx

name = "100Epochs"
# Load the Keras model from the .h5 file
#model = tf.keras.models.load_model('../saved_models/100Epochs/new_out_model_best.weights.h5')
#model = tf.keras.models.load_model('new_out_model_best.weights.h5')
model = tf.keras.models.load_model('new_out_model_best.h5')


print(model.inputs[0].shape,"-shape")
# Specify the input shape of the model
input_signature = [tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name='input')]

# Convert the model to ONNX
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)

# Save the ONNX model to a file
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

# Verify the ONNX model
onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)
print(onnx.helper.printable_graph(onnx_model.graph))