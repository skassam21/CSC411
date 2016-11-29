import random
import time

import math
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
IMAGE_SIZE = 64
INPUT_SIZE = 64 * 64 * 3
IMAGE_RED = 0.5
NUM_LABELS = 8
VALIDATION_PERCENTAGE = 0.2
VALIDATION_SIZE = 1000  # Size of the validation set.
BATCH_SIZE = 300
NUM_EPOCHS = 3000
EVAL_BATCH_SIZE = 1000
EVAL_FREQUENCY = 100  # Number of steps between evaluations.
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
    data = np.zeros((end - begin, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    for i, filename in enumerate(filenames):
        image = ndimage.imread(filename, flatten=False)
        image = misc.imresize(image, IMAGE_RED).reshape(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
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


def randomly_flip_images(d, l, likely_to_flip=0.5):
    final_d = []
    labels = []
    for i in range(d.shape[0]):
        if random.random() <= likely_to_flip:
            final_d.append(d[i, :, ::-1, :])
            labels.append(l[i])
        final_d.append(d[i, :, :, :])
        labels.append(l[i])
    all_data = np.zeros((len(final_d), d.shape[1], d.shape[2], d.shape[3]))
    for i in range(len(final_d)):
        all_data[i, :, :, :] = final_d[i]
    return all_data, np.array(labels)


def flip_all(x, td_shape):
    train_data = x[:, :-1]
    labels = x[:, -1]
    print labels.shape
    print train_data.shape
    train_data = train_data.reshape((x.shape[0], td_shape[1], td_shape[2], td_shape[3]))
    all_train_data = np.concatenate((train_data, train_data[:, :, ::-1, :]))
    all_labels = np.concatenate((labels, labels))
    print all_train_data.shape
    print all_labels.shape
    return convert_to_one_arr(all_train_data, all_labels)


def convert_to_one_arr(train_data, train_labels):
    td_shape = train_data.shape
    reshaped_data = train_data.reshape((td_shape[0], td_shape[1] * td_shape[2] * td_shape[3]))
    all_data = np.zeros((td_shape[0],  td_shape[1] * td_shape[2] * td_shape[3] + 1))
    all_data[:, :-1] = reshaped_data
    all_data[:, -1] = train_labels
    return all_data


def shuffle_data_2(train_data, train_labels, percentage_validation, number_classes, threshold=1000):
    td_shape = train_data.shape
    all_data = convert_to_one_arr(train_data, train_labels)
    all_data_ordered = all_data[all_data[:, -1].argsort()]
    training_data = []
    validation_data = []
    for i in range(number_classes):
        x = all_data_ordered[np.where(all_data_ordered[:, -1] == i)[0]]
        np.random.shuffle(x)
        if x.shape[0] <= threshold:
            print "shape of class {} is {}".format(i, x.shape[0])
            x = flip_all(x, td_shape)
            print "shape of class {} is {} after flipping".format(i, x.shape[0])
        number_validation = int(math.floor(x.shape[0] * percentage_validation))
        validation_data.append(x[:number_validation])
        training_data.append(x[number_validation:])

    training_data = np.concatenate(([t for t in training_data]))
    validation_data = np.concatenate(([v for v in validation_data]))
    np.random.shuffle(training_data)
    np.random.shuffle(validation_data)
    return validation_data[:, :-1].reshape((validation_data.shape[0], td_shape[1], td_shape[2], td_shape[3])),\
           np.array(validation_data[:, -1], dtype=np.int64), \
           training_data[:, :-1].reshape((training_data.shape[0], td_shape[1], td_shape[2], td_shape[3])),\
           np.array(training_data[:, -1], dtype=np.int64)


def rotate_images(train_data, train_labels):
    td_shape = train_data.shape
    all_data = np.zeros((td_shape[0] * 4, td_shape[1], td_shape[2], td_shape[3]))
    train_data_rotated_90 = np.transpose(np.rot90(np.transpose(train_data, [1, 2, 0, 3])), [2, 0, 1, 3])
    train_data_rotated_180 = np.transpose(np.rot90(np.transpose(train_data, [1, 2, 0, 3]), 2), [2, 0, 1, 3])
    train_data_rotated_270 = np.transpose(np.rot90(np.transpose(train_data, [1, 2, 0, 3]), 3), [2, 0, 1, 3])
    all_data[:td_shape[0], :, :, :] = train_data
    all_data[td_shape[0]:td_shape[0]*2, :, :, :] = train_data_rotated_90
    all_data[td_shape[0]*2:td_shape[0]*3, :, :, :] = train_data_rotated_180
    all_data[td_shape[0]*3:td_shape[0]*4, :, :, :] = train_data_rotated_270
    all_labels = np.concatenate((train_labels, train_labels, train_labels, train_labels))
    return all_data, all_labels


def DisplayPlot(train, valid, ylabel, number=0):
    """Displays training curve.

    Args:
        train: Training statistics.
        valid: Validation statistics.
        ylabel: Y-axis label of the plot.
    """
    plt.figure(number)
    plt.clf()
    train = np.array(train)
    valid = np.array(valid)
    plt.plot(train[:, 0], train[:, 1], 'b', label='Train')
    plt.plot(valid[:, 0], valid[:, 1], 'g', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.draw()
    plt.pause(0.0001)


def prepare_data_for_nn(d):
    return d.reshape(d.shape[0], d.shape[1] * d.shape[2] * d.shape[3])


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

        validation_data, validation_labels, \
            train_data, train_labels = shuffle_data_2(train_data, train_labels, VALIDATION_PERCENTAGE, NUM_LABELS)
        validation_data = prepare_data_for_nn(validation_data)
        train_data = prepare_data_for_nn(train_data)
        print train_data.shape
        print train_labels.shape
        # Generate a validation set.
        # validation_data = train_data[:VALIDATION_SIZE, ...]
        # validation_labels = train_labels[:VALIDATION_SIZE]
        # train_data = train_data[VALIDATION_SIZE:, ...]
        # train_labels = train_labels[VALIDATION_SIZE:]
        # train_data, train_labels = rotate_images(train_data, train_labels)
        num_epochs = NUM_EPOCHS
    train_size = train_labels.shape[0]
    print(train_labels.shape)
    print(train_data.shape)
    print(validation_data.shape)


    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    train_data_node = tf.placeholder(
            data_type(),
            shape=(BATCH_SIZE, INPUT_SIZE))
    train_labels_node = tf.placeholder(np.int64, shape=(BATCH_SIZE,))
    eval_data = tf.placeholder(
            data_type(),
            shape=(EVAL_BATCH_SIZE, INPUT_SIZE))

    # The variables below hold all the trainable weights. They are passed an
    # initial value which will be assigned when we call:
    # {tf.initialize_all_variables().run()}
    fc1_weights = tf.Variable(  # fully connected, depth 512.
            tf.truncated_normal([INPUT_SIZE, 16],
                                stddev=0.1,
                                seed=SEED,
                                dtype=data_type()))
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[16], dtype=data_type()))
    fc2_weights = tf.Variable(  # fully connected, depth 100.
            tf.truncated_normal([16, 32],
                                stddev=0.1,
                                seed=SEED,
                                dtype=data_type()))
    fc2_biases = tf.Variable(tf.constant(0.1, shape=[32], dtype=data_type()))
    fc3_weights = tf.Variable(  # fully connected, depth 100.
            tf.truncated_normal([32, 64],
                                stddev=0.1,
                                seed=SEED,
                                dtype=data_type()))
    fc3_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=data_type()))
    fc4_weights = tf.Variable(tf.truncated_normal([64, NUM_LABELS],
                                                  stddev=0.1,
                                                  seed=SEED,
                                                  dtype=data_type()))
    fc4_biases = tf.Variable(tf.constant(
            0.1, shape=[NUM_LABELS], dtype=data_type()))

    # We will replicate the model structure for the training subgraph, as well
    # as the evaluation subgraphs, while sharing the trainable parameters.
    def model(data, train=False):
        """The Model definition."""
        # 2D convolution, with 'SAME' padding (i.e. the output feature map has
        # the same size as the input). Note that {strides} is a 4D array whose
        # shape matches the data layout: [image index, y, x, depth].
        hidden = tf.nn.relu(tf.matmul(data, fc1_weights) + fc1_biases)
        hidden2 = tf.nn.relu(tf.matmul(hidden, fc2_weights) + fc2_biases)
        hidden3 = tf.nn.relu(tf.matmul(hidden2, fc3_weights) + fc3_biases)

        # Add a 50% dropout during training only. Dropout also scales
        # activations such that no rescaling is needed at evaluation time.
        # if train:
        #     hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
        return tf.matmul(hidden3, fc4_weights) + fc4_biases

    # Training computation: logits + cross-entropy loss.
    logits = model(train_data_node, True)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits, train_labels_node))

    # # L2 regularization for the fully connected parameters.
    # regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
    #                 tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
    # # Add the regularization term to the loss.
    # loss += 5e-4 * regularizers

    # Optimizer: set up a variable that's incremented once per batch and
    # controls the learning rate decay.
    batch = tf.Variable(0, dtype=data_type())
    # Decay once per epoch, using an exponential schedule starting at 0.01.
    learning_rate = tf.train.exponential_decay(
            0.01,  # Base learning rate.
            batch * BATCH_SIZE,  # Current index into the dataset.
            train_size,  # Decay step.
            1.0,  # Decay rate.
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

    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Run all the initializers to prepare the trainable parameters.
        tf.initialize_all_variables().run()
        print('Initialized!')
        train_error_list = []
        valid_error_list = []
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
                save_path = saver.save(sess, "nn/model-16-32-64-{}.ckpt".format(step))
                print("Model saved in file: %s" % save_path)
                elapsed_time = time.time() - start_time
                start_time = time.time()
                print('Step %d (epoch %.2f), %.1f ms' %
                      (step, float(step) * BATCH_SIZE / train_size,
                       1000 * elapsed_time / EVAL_FREQUENCY))
                validation_error = error_rate(
                        eval_in_batches(validation_data, sess), validation_labels)
                train_error = error_rate(predictions, batch_labels)
                print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                print('Minibatch error: %.1f%%' % train_error)
                print('Validation error: %.1f%%' % validation_error)
                train_error_list.append((step, train_error))
                valid_error_list.append((step, validation_error))
                DisplayPlot(train_error_list, valid_error_list, 'Error', number=0)
                sys.stdout.flush()
        # Finally print the result!
        # test_error = error_rate(eval_in_batches(test_data, sess), test_labels)
        # print('Test error: %.1f%%' % test_error)
        # if FLAGS.self_test:
        #     print('test_error', test_error)
        #     assert test_error == 0.0, 'expected 0.0 test_error, got %.2f' % (
        #         test_error,)
        raw_input('Press Enter to continue.')


if __name__ == '__main__':
    tf.app.run()
