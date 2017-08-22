#! /usr/bin/env python3

import tensorflow as tf
import numpy as np
from PIL import Image
import os
import glob

def load_image(file_path, is_resize=True, width=48, height=72):
    image = Image.open(file_path).convert('RGB')
    if is_resize:
        image = image.resize((width, height))
    return np.array(image)

def get_samples(folder, label, is_resize=True, width=48, height=72):
    imgs = []
    labels = []
    file_paths = os.path.join(folder, '*.jpg')
    for file_path in sorted(glob.glob(file_paths)):
        img = load_image(file_path, is_resize, width, height)
        imgs.append(img)
        labels.append(label)

    return np.asarray(imgs), np.asarray(labels)

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_records(writer, folder, label, is_resize=True, width=48, height=72):
    file_paths = os.path.join(folder, '*.jpg')
    for file_path in sorted(glob.glob(file_paths)):
        image = load_image(file_path, is_resize, width, height)
        image_raw = image.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': _int64_feature(label),
            'img_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())

def count_records(filenames):
    count = 0
    if isinstance(filenames, str):
        filenames = [filenames]

    for filename in filenames:
        for record in tf.python_io.tf_record_iterator(filename):
            count += 1

    return count

def read_and_decode(filename_queue, height, width):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features={
                       'label': tf.FixedLenFeature([], tf.int64),
                       'img_raw' : tf.FixedLenFeature([], tf.string),
                   })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [height, width, 3])
    try: # TF12
        img = tf.image.per_image_standardization(img)
    except: #earlier TF versions
        img = tf.image.per_image_whitening(img)
    # process

    label = tf.cast(features['label'], tf.int32)
    return img, label

def batch(filename, is_train, height, width, batch_size):
    filename_queue = tf.train.string_input_producer([filename])
    example, label = read_and_decode(filename_queue, height, width)
    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * batch_size
    if is_train:
        example_batch, label_batch = tf.train.shuffle_batch(
            [example, label], batch_size=batch_size, capacity=capacity,
            min_after_dequeue=min_after_dequeue, num_threads=1)
    else:
        example_batch, label_batch = tf.train.batch(
            [example, label], batch_size=batch_size, capacity=capacity, num_threads=1)
    return example_batch, label_batch

    # filename_queue = tf.train.string_input_producer([filename], shuffle=True)
    # read_threads = 10
    # example_list = [read_and_decode(filename_queue, height, width) for _ in range(read_threads)]
    # min_after_dequeue = 1000
    # capacity = min_after_dequeue + 3 * batch_size
    # img_batch, label_batch = tf.train.shuffle_batch_join(example_list,
    #                                             batch_size=batch_size,
    #                                             capacity=capacity,
    #                                             min_after_dequeue=min_after_dequeue)

    # return img_batch, label_batch

