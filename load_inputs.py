import os
import csv
from scipy import ndimage

import numpy as np
import tensorflow as tf
from scipy import misc


def extract_data(begin, end):
    """Extract the images into a 4D tensor [image index, y, x, channels].
  """
    filenames = [os.path.join('411a3/train/%05d.jpg' % i) for i in xrange(begin + 1, end + 1)]

    data = np.zeros((end - begin, 64, 64, 3))
    for i, filename in enumerate(filenames):
        image = ndimage.imread(filename, flatten=False)
        image = misc.imresize(image, 0.5)
        # image = image.reshape(64, 64, 1)
        # imgplot = plt.imshow(image)
        # plt.pause(10000)
        data[i] = image
    return tf.to_float(data)


def extract_labels(begin, end):
    with open('411a3/train.csv', 'rb') as f:
        reader = csv.reader(f)
        labels = list(reader)
    labels = [int(label[1]) - 1 for label in labels[begin:end]]
    return np.array(labels)


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    """Construct a queued batch of images and labels.

    Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

    Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
                [image, label],
                batch_size=batch_size,
                enqueue_many=True,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * batch_size,
                min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
                [image, label],
                enqueue_many=True,
                batch_size=batch_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    tf.image_summary('images', images)
    print label_batch
    return images, tf.reshape(label_batch, [batch_size])


def get_inputs(batch_size):
    train_data = extract_data(0, 7000)
    train_labels = extract_labels(0, 7000)

    return _generate_image_and_label_batch(train_data, train_labels,
                                           int(50000 * 0.4), batch_size,
                                           shuffle=True)
