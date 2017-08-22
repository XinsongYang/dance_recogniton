import os
import glob
import numpy as np
from PIL import Image
from sklearn.utils import shuffle

class DataSet(object):

    def __init__(self, images, labels):
        self._num_examples = images.shape[0]

        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)
        images = images.reshape(self._num_examples, -1)

        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

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

        return self._images[start:end], self._labels[start:end]

def video_to_img_set(videos, labels, frame_size, width, height, channel_num=3):
    images = videos.reshape(-1, height, width, channel_num)
    img_labels = []
    for label in labels:
        img_labels += [label] * frame_size
    img_labels = np.array(img_labels)
    num_examples = images.shape[0]
    perm = np.arange(num_examples)
    np.random.shuffle(perm)
    images = images[perm]
    img_labels = img_labels[perm]
    return images, img_labels

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

    train_imgs, train_img_labels = video_to_img_set(train_videos, train_labels, frame_size, width, height)
    test_imgs, test_img_labels = video_to_img_set(test_videos, test_labels, frame_size, width, height)

    return DataSet(train_imgs, train_img_labels), DataSet(test_imgs, test_img_labels)

# train_data, test_data = load_data('../img', ['dance', 'nodance'], 11, 40, 72, 3)
# print(train_data.images.shape, train_data.labels.shape)
