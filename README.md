# Trained CNNs

This repository containes a collection of CNNs trained on different image sets.

The CNNs are contained within two files:

- `frozen_model.pb`: The trained CNN frozen into Tensorflow protobuf format.
- `network_info.xml`: Information about the network and the input tensors.

Most of the networks were trained with Tensorflow 1.14.

The easiest way to use them for classification is via our _ParticleTrieur_ software which can be downloaded from http://particle-classification.readthedocs.io

Alternatively, to use the models in python:

1. Read the `network_info.xml` file to find the image width, height and number of channels (1 = greyscale, 3 = colour), and the names of the input and output tensors.

2. Load your images and resize them to this format (converting to greyscale if necessary). If the images are not square, pad them symmetrically to square using the median value of the border pixels.

3. Rescale the image intensity to the range \[0,1\], e.g. for 8 bit images divide by 255.

The prediction scores and class values can then be found using our MISO library by:
```python
import tensorflow.keras.backend as K

# Load graph
source = r'C:/Path/To/The/CNN/frozen_model.pb'
session = K.get_session()
with gfile.Open(source, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    session.graph.as_default()
    tf.import_graph_def(graph_def, name='')

# Find input and output tensors
input_tensor = 'input_1:0'
output_tensor = 'dense_1/Softmax:0'
input = session.graph.get_tensor_by_name(input_tensor)
output = session.graph.get_tensor_by_name(output_tensor)
    
# Calculate predictions
batch_size = 64
preds = []
for i in range(0, len(images), batch_size):
    preds.append(session.run(output, feed_dict={input: images[i:i+batch_size]}))
preds = np.concatenate(preds, axis=0)
pred_cls = np.argmax(preds, axis=1)
```
where `input_tensor` and `output_tensor` are the operation names of input and output (preds) tensors described in the `network_info.xml` file with `:0` appended, and `source` is the path to the `frozen_model.pb` file.
