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
IMAGE_SIZE = 128
NUM_LABELS = 8
VALIDATION_SIZE = 1000  # Size of the validation set.
BATCH_SIZE = 300
NUM_EPOCHS = 3000
EVAL_BATCH_SIZE = 300
EVAL_FREQUENCY = 10  # Number of steps between evaluations.
SEED = 10
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
    print np.argmax(predictions, 1)[1:10]
    print labels[1:10]
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


def main(argv=None):  # pylint: disable=unused-argument
    if FLAGS.self_test:
        print('Running self-test.')
        train_data, train_labels = fake_data(256)
        validation_data, validation_labels = fake_data(EVAL_BATCH_SIZE)
        test_data, test_labels = fake_data(EVAL_BATCH_SIZE)
        num_epochs = 1
    else:
        # Get the data.
        # Extract it into numpy arrays.
        train_data = extract_data(0, 7000)
        train_labels = extract_labels(0, 7000)
        test_data = extract_data(100, 200)
        test_labels = extract_labels(100, 200)

        # Generate a validation set.
        validation_data = train_data[:VALIDATION_SIZE, ...]
        validation_labels = train_labels[:VALIDATION_SIZE]
        train_data = train_data[VALIDATION_SIZE:, ...]
        train_labels = train_labels[VALIDATION_SIZE:]
        num_epochs = NUM_EPOCHS
    train_size = train_labels.shape[0]

    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    train_data_node = tf.placeholder(
            data_type(),
            shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    train_labels_node = tf.placeholder(np.int64, shape=(BATCH_SIZE,))
    eval_data = tf.placeholder(
            data_type(),
            shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

    # The variables below hold all the trainable weights. They are passed an
    # initial value which will be assigned when we call:
    # {tf.initialize_all_variables().run()}
    conv1_weights = tf.Variable(
            tf.truncated_normal([5, 5, NUM_CHANNELS, 15],  # 5x5 filter, depth 8.
                                stddev=0.1,
                                seed=SEED, dtype=data_type()))
    conv1_biases = tf.Variable(tf.zeros([15], dtype=data_type()))
    conv2_weights = tf.Variable(tf.truncated_normal(
            [5, 5, 15, 30], stddev=0.1,
            seed=SEED, dtype=data_type()))
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[30], dtype=data_type()))
    conv3_weights = tf.Variable(tf.truncated_normal(
            [5, 5, 30, 60], stddev=0.1,
            seed=SEED, dtype=data_type()))
    conv3_biases = tf.Variable(tf.constant(0.1, shape=[60], dtype=data_type()))
    fc1_weights = tf.Variable(  # fully connected, depth 512.
            tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 60, 150],
                                stddev=0.1,
                                seed=SEED,
                                dtype=data_type()))
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[150], dtype=data_type()))
    fc2_weights = tf.Variable(tf.truncated_normal([150, NUM_LABELS],
                                                  stddev=0.1,
                                                  seed=SEED,
                                                  dtype=data_type()))
    fc2_biases = tf.Variable(tf.constant(
            0.1, shape=[NUM_LABELS], dtype=data_type()))

    # We will replicate the model structure for the training subgraph, as well
    # as the evaluation subgraphs, while sharing the trainable parameters.
    def model(data, train=False):
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
        conv = tf.nn.conv2d(pool,
                            conv3_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv3_biases))

        # Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
        pool_shape = relu.get_shape().as_list()
        reshape = tf.reshape(
                relu,
                [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
        # Add a 50% dropout during training only. Dropout also scales
        # activations such that no rescaling is needed at evaluation time.
        if train:
            hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
        return tf.matmul(hidden, fc2_weights) + fc2_biases

    # Training computation: logits + cross-entropy loss.
    logits = model(train_data_node, True)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits, train_labels_node))

    # L2 regularization for the fully connected parameters.
    # regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
    #                 tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
    # Add the regularization term to the loss.
    # loss += 5e-4 * regularizers

    # Optimizer: set up a variable that's incremented once per batch and
    # controls the learning rate decay.
    batch = tf.Variable(0, dtype=data_type())
    # Decay once per epoch, using an exponential schedule starting at 0.01.
    learning_rate = tf.train.exponential_decay(
            0.01,  # Base learning rate.
            batch * BATCH_SIZE,  # Current index into the dataset.
            train_size,  # Decay step.
            0.95,  # Decay rate.
            staircase=True)
    # Use simple momentum for the optimization.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=batch)

    # Predictions for the current training minibatch.
    train_prediction = tf.nn.softmax(logits)

    # Predictions for the test and validation, which we'll compute less often.
    eval_prediction = tf.nn.softmax(model(eval_data))

    # Small utility function to evaluate a dataset by feeding batches of data to
    # {eval_data} and pulling the results from {eval_predictions}.
    # Saves memory and enables this to run on smaller GPUs.
    def eval_in_batches(data, sess):
        """Get all predictions for a dataset by running it in small batches."""
        size = data.shape[0]
        if size < EVAL_BATCH_SIZE:
            raise ValueError("batch size for evals larger than dataset: %d" % size)
        predictions = np.ndarray(shape=(size, NUM_LABELS), dtype=np.float32)
        for begin in xrange(0, size, EVAL_BATCH_SIZE):
            end = begin + EVAL_BATCH_SIZE
            if end <= size:
                predictions[begin:end, :] = sess.run(
                        eval_prediction,
                        feed_dict={eval_data: data[begin:end, ...]})
            else:
                batch_predictions = sess.run(
                        eval_prediction,
                        feed_dict={eval_data: data[-EVAL_BATCH_SIZE:, ...]})
                predictions[begin:, :] = batch_predictions[begin - size:, :]
        return predictions

    # Create a local session to run the training.
    start_time = time.time()
    with tf.Session() as sess:
        # Run all the initializers to prepare the trainable parameters.
        tf.initialize_all_variables().run()
        print('Initialized!')
        # Loop through training steps.
        for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
            # Compute the offset of the current minibatch in the data.
            # Note that we could use better randomization across epochs.
            offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
            batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
            batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
            # This dictionary maps the batch data (as a numpy array) to the
            # node in the graph it should be fed to.
            feed_dict = {train_data_node: batch_data,
                         train_labels_node: batch_labels}
            # Run the graph and fetch some of the nodes.
            _, l, lr, predictions = sess.run(
                    [optimizer, loss, learning_rate, train_prediction],
                    feed_dict=feed_dict)
            print step
            if step % EVAL_FREQUENCY == 0:
                elapsed_time = time.time() - start_time
                start_time = time.time()
                print('Step %d (epoch %.2f), %.1f ms' %
                      (step, float(step) * BATCH_SIZE / train_size,
                       1000 * elapsed_time / EVAL_FREQUENCY))
                print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                print('Minibatch error: %.1f%%' % error_rate(predictions, batch_labels))
                print('Validation error: %.1f%%' % error_rate(
                        eval_in_batches(validation_data, sess), validation_labels))
                sys.stdout.flush()
        # Finally print the result!
        test_error = error_rate(eval_in_batches(test_data, sess), test_labels)
        print('Test error: %.1f%%' % test_error)
        if FLAGS.self_test:
            print('test_error', test_error)
            assert test_error == 0.0, 'expected 0.0 test_error, got %.2f' % (
                test_error,)


if __name__ == '__main__':
    tf.app.run()
