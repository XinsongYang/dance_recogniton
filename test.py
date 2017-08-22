import os
import glob
import numpy as np
from PIL import Image
from sklearn.utils import shuffle
import tensorflow as tf

data_path = './test_data_20170714'
classes = ['dance', 'nodance']
classes_num = len(classes)
frame_size = 7
img_width = 48
img_height = 72
channel_num = 3

def load_test_data(path, classes, width, height, channel_num=3, ratio=0.2):
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
            labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    num_examples = images.shape[0]
    perm = np.arange(num_examples)
    np.random.shuffle(perm)
    images = images[perm]
    labels = labels[perm]

    images = images.astype(np.float32)
    images = np.multiply(images, 1.0 / 255.0)
    images = images.reshape(num_examples, -1)

    num_train = np.int(num_examples * ratio)
    test_images = images[0: num_train - 1]
    test_labels = labels[0: num_train - 1]

    return test_images, test_labels


test_images, test_labels = load_test_data(data_path, classes, img_width, img_height, channel_num)

sess=tf.Session()    

saver = tf.train.import_meta_graph('model/single/bn_single-5000.meta')
saver.restore(sess,tf.train.latest_checkpoint('model/single/'))

graph = tf.get_default_graph()
x = graph.get_tensor_by_name("x:0")
y_ = graph.get_tensor_by_name("y_:0")
keep_prob = graph.get_tensor_by_name("keep_prob:0")
phase = graph.get_tensor_by_name("phase:0")
y_conv = graph.get_tensor_by_name("y_conv:0")
feed_dict = {x: test_images, y_: test_labels, keep_prob: 1.0, phase: False}

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict))