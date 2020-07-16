# Just disables the warning, doesn't enable AVX/FMA
import os
import tensorflow.compat.v1 as tf
import numpy as np
import PIL
# from nets.mobilenet import mobilenet_v2

# suppress warning about AVX/FMA instruction set for CPU (this app does not use cpu)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

img = np.array(PIL.Image.open('panda.jpeg').resize((224, 224))).astype(np.float) / 128 - 1
gd = tf.GraphDef.FromString(open('mobilenet_v2_0.35_96' + '_frozen.pb', 'rb').read())
inp, predictions = tf.import_graph_def(gd,  return_elements = ['input:0', 'MobilenetV2/Predictions/Reshape_1:0'])


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph


def main():
    pom = load_graph('mobilenet_v2_0.35_96_frozen.pb')
    pass


if __name__ == '__main__':
    # cProfile.run(main(sys.argv[1:]))
    main()
