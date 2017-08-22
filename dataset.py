import os
import glob
import numpy as np
from PIL import Image
from sklearn.utils import shuffle

class DataSet(object):

    def __init__(self, videos, labels):
        self._num_examples = videos.shape[0]

        videos = videos.astype(np.float32)
        videos = np.multiply(videos, 1.0 / 255.0)
        videos = videos.reshape(self._num_examples, -1)

        self._videos = videos
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def videos(self):
        return self._videos

    @property
    def labels(self):
        return self._labels

    @property
    def num(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return self._videos[start:end], self._labels[start:end]

def load_data(path, classes, frame_size, width, height, channel_num=3, train_ratio=0.8):
    images = []
    labels = []

    print('Reading training images')
    for class_name in classes:
        index = classes.index(class_name)
        print('Loading {} files (Index: {})'.format(class_name, index))
        file_path = os.path.join(path, class_name, '*.jpg')
        for file in sorted(glob.glob(file_path)):
            image = Image.open(file)
            image = np.array(image.resize((width, height)))
            images.append(image)
        label = [0] * len(classes)
        label[index] = 1.0
        labels += [label] * int(len(images) / frame_size)

    images = np.array(images)
    videos = images.reshape(-1, frame_size, height, width, channel_num)
    labels = np.array(labels)

    num_examples = videos.shape[0]
    perm = np.arange(num_examples)
    np.random.shuffle(perm)
    videos = videos[perm]
    labels = labels[perm]

    num_train = np.int(num_examples * train_ratio)
    train_videos = videos[0: num_train - 1]
    train_labels = labels[0: num_train - 1]
    test_videos = videos[num_train: -1]
    test_labels = labels[num_train: -1]

    return DataSet(train_videos, train_labels), DataSet(test_videos, test_labels)

# train_data, test_data = load_data('../img', ['dance', 'nodance'], 11, 40, 72, 3)
