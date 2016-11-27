import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import sys
import numpy as np
from scipy import ndimage
from scipy import misc
import os
import csv
import tensorflow as tf

NUM_CHANNELS = 3
IMAGE_SIZE = 32
NUM_LABELS = 8
VALIDATION_SIZE = 1000  # Size of the validation set.
BATCH_SIZE = 300
NUM_EPOCHS = 3000
EVAL_BATCH_SIZE = 300
EVAL_FREQUENCY = 10  # Number of steps between evaluations.
SEED = 66478  # Set to None for random seed.
PIXEL_DEPTH = 255

tf.app.flags.DEFINE_boolean("self_test", False, "True if running a self test.")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            "Use half floats instead of full floats if True.")
tf.app.flags.DEFINE_string('train_dir', 'train_tensorboard/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
FLAGS = tf.app.flags.FLAGS


def extract_data(begin, end):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
  """
    filenames = [os.path.join('411a3/train/%05d.jpg' % i) for i in xrange(begin + 1, end + 1)]
    data = np.zeros((end - begin, IMAGE_SIZE, IMAGE_SIZE, 3))
    for i, filename in enumerate(filenames):
        image = ndimage.imread(filename, flatten=False)
        image = misc.imresize(image, 0.25)
        data[i, :, :, :] = image
        data[i, :, :, :] = (data[i, :, :, :] - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    if FLAGS.use_fp16:
        data = np.array(data, dtype=np.float16)
    else:
        data = np.array(data, dtype=np.float32)
    return data

def extract_test_data(begin, end):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
  """
    filenames = [os.path.join('411a3/val/%05d.jpg' % i) for i in xrange(begin + 1, end + 1)]
    data = np.zeros((end - begin, IMAGE_SIZE, IMAGE_SIZE, 3))
    for i, filename in enumerate(filenames):
        image = ndimage.imread(filename, flatten=False)
        image = misc.imresize(image, 0.25)
        data[i, :, :, :] = image
        data[i, :, :, :] = (data[i, :, :, :] - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    if FLAGS.use_fp16:
        data = np.array(data, dtype=np.float16)
    else:
        data = np.array(data, dtype=np.float32)
    return data


def extract_labels(begin, end):
    with open('411a3/train.csv', 'rb') as f:
        reader = csv.reader(f)
        labels = list(reader)
    labels = [int(label[1]) - 1 for label in labels[begin:end]]
    return np.array(labels)


def data_type():
    """Return the type of the activations, weights, and placeholder variables."""
    if FLAGS.use_fp16:
        return np.float16
    else:
        return np.float32


def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and sparse labels."""
    print np.argmax(predictions, 1)[1:100]
    print labels[1:100]
    return 100.0 - (
        100.0 *
        np.sum(np.argmax(predictions, 1) == labels) /
        predictions.shape[0])


def fake_data(num_images):
    """Generate a fake dataset that matches the dimensions of MNIST."""
    data = np.ndarray(
            shape=(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS),
            dtype=np.float32)
    labels = np.zeros(shape=(num_images,), dtype=np.int64)
    for image in xrange(num_images):
        label = image % 2
        data[image, :, :, 0] = label - 0.5
        labels[image] = label
    return data, labels


def shuffle_data(train_data, train_labels):
    td_shape = train_data.shape
    reshaped_data = train_data.reshape((td_shape[0], td_shape[1] * td_shape[2] * td_shape[3]))
    all_data = np.zeros((td_shape[0],  td_shape[1] * td_shape[2] * td_shape[3] + 1))
    all_data[:, :-1] = reshaped_data
    all_data[:, -1] = train_labels
    np.random.shuffle(all_data)
    return all_data[:, :-1].reshape((td_shape[0], td_shape[1], td_shape[2], td_shape[3])), \
           np.array(all_data[:, -1], dtype=np.int64)


# We will replicate the model structure for the training subgraph, as well
    # as the evaluation subgraphs, while sharing the trainable parameters.
def model(data, conv1_weights, conv1_biases, conv2_weights, conv2_biases,
          fc1_weights, fc1_biases, fc2_weights, fc2_biases, train=False):
    """The Model definition."""
    # 2D convolution, with 'SAME' padding (i.e. the output feature map has
    # the same size as the input). Note that {strides} is a 4D array whose
    # shape matches the data layout: [image index, y, x, depth].
    conv = tf.nn.conv2d(data,
                        conv1_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    # Bias and rectified linear non-linearity.
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
    # Max pooling. The kernel size spec {ksize} also follows the layout of
    # the data. Here we have a pooling window of 2, and a stride of 2.
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    conv = tf.nn.conv2d(pool,
                        conv2_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')

    # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # fully connected layers.
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(
            pool,
            [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
    # Fully connected layer. Note that the '+' operation automatically
    # broadcasts the biases.
    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
    # Add a 50% dropout during training only. Dropout also scales
    # activations such that no rescaling is needed at evaluation time.
    if train:
        hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
    return tf.matmul(hidden, fc2_weights) + fc2_biases

def save_data(eval_labels):
    with open('labels_test.csv', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow(["Id", "Prediction"])
        i = 1
        for data in eval_labels:
            spamwriter.writerow([i, data])
            i += 1
        for i in range(971, 2971):
            spamwriter.writerow([i, 0])


def main(argv=None):  # pylint: disable=unused-argument
    test_data = extract_test_data(0, 970)

    eval_data = tf.placeholder(
            data_type(),
            shape=(test_data.shape[0], IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

    # Predictions for the test and validation, which we'll compute less often.
    # Small utility function to evaluate a dataset by feeding batches of data to
    # {eval_data} and pulling the results from {eval_predictions}.
    # Saves memory and enables this to run on smaller GPUs.
    def eval_in_batches(data, sess):
        """Get all predictions for a dataset by running it in small batches."""
        batch_predictions = sess.run(
                eval_prediction,
                feed_dict={eval_data: data})
        return np.argmax(batch_predictions, 1) + 1

    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph('a3/model.ckpt.meta')
        new_saver.restore(sess, 'a3/model.ckpt')

        # The variables below hold all the trainable weights. They are passed an
        # initial value which will be assigned when we call:
        # {tf.initialize_all_variables().run()}
        conv1_weights = tf.trainable_variables()[0].value()
        conv1_biases = tf.trainable_variables()[1].value()
        conv2_weights = tf.trainable_variables()[2].value()
        conv2_biases = tf.trainable_variables()[3].value()
        fc1_weights = tf.trainable_variables()[4].value()
        fc1_biases = tf.trainable_variables()[5].value()
        fc2_weights = tf.trainable_variables()[6].value()
        fc2_biases = tf.trainable_variables()[7].value()
        # Run all the initializers to prepare the trainable parameters.
        eval_prediction = tf.nn.softmax(model(eval_data, conv1_weights, conv1_biases, conv2_weights,
                                              conv2_biases, fc1_weights, fc1_biases, fc2_weights, fc2_biases))

        save_data(eval_in_batches(test_data, sess))


if __name__ == '__main__':
    tf.app.run()
