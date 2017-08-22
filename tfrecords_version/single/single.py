import tensorflow as tf
import numpy as np
import data_process as dp
import math
import time

data_path = './dance_data'
classes = ['dance', 'undance']
classes_num = len(classes)
frame_size = 11
img_width = 48
img_height = 72
channel_num = 3
train_ratio = 0.8
img_size_flat = img_width * img_height * channel_num

# Convolutional Layer 1.
filter_num1 = 96
filter_size1 = 11 
stride1 = 3

# Convolutional Layer 2.
filter_num2 = 256
filter_size2 = 5
stride2 = 1

# Convolutional Layer 3.
filter_num3 = 384
filter_size3 = 3
stride3 = 1

# Convolutional Layer 4.
filter_num4 = 384
filter_size4 = 3
stride4 = 1

# Convolutional Layer 5.
filter_num5 = 256
filter_size5 = 3
stride5 = 1

flat_size = 3 * 2 * filter_num5

# Fully connected layer
full_connect_num1 = 4096
full_connect_num2 = 4096

# train_data, test_data = single_dataset.load_data(data_path, classes, frame_size, img_width, img_height, channel_num, train_ratio)

def inference(x):

    x_image = tf.reshape(x, [-1, img_height, img_width, channel_num])

    # First convolutional layer
    W_conv1 = weight_variable([filter_size1, filter_size1, channel_num, filter_num1])
    b_conv1 = bias_variable([filter_num1])
    conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, stride1, stride1, 1], padding='SAME')
    h_conv1 = tf.nn.relu(conv1 + b_conv1)

    # # Normalization layer
    h_normal1 = tf.nn.local_response_normalization(h_conv1, 5, 2, 1e-4, 0.5)

    # Pooling layer 
    h_pool1 = max_pool_2x2(h_normal1)

    # Second convolutional layer
    W_conv2 = weight_variable([filter_size2, filter_size2, filter_num1, filter_num2])
    b_conv2 = bias_variable([filter_num2])
    conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1, stride2, stride2, 1], padding='SAME')
    h_conv2 = tf.nn.relu(conv2 + b_conv2)

    # Normalization layer
    h_normal2 = tf.nn.local_response_normalization(h_conv2, 5, 2, 1e-4, 0.5)

    # Second pooling layer.
    h_pool2 = max_pool_2x2(h_normal2)

    # 3 convolutional layer
    W_conv3 = weight_variable([filter_size3, filter_size3, filter_num2, filter_num3])
    b_conv3 = bias_variable([filter_num3])
    conv3 = tf.nn.conv2d(h_pool2, W_conv3, strides=[1, stride3, stride3, 1], padding='SAME')
    h_conv3 = tf.nn.relu(conv3 + b_conv3)

    # 4 convolutional layer
    W_conv4 = weight_variable([filter_size4, filter_size4, filter_num3, filter_num4])
    b_conv4 = bias_variable([filter_num4])
    conv4 = tf.nn.conv2d(h_conv3, W_conv4, strides=[1, stride4, stride4, 1], padding='SAME')
    h_conv4 = tf.nn.relu(conv4 + b_conv4)

    # 5 convolutional layer
    W_conv5 = weight_variable([filter_size5, filter_size5, filter_num4, filter_num5])
    b_conv5 = bias_variable([filter_num5])
    conv5 = tf.nn.conv2d(h_conv4, W_conv5, strides=[1, stride5, stride5, 1], padding='SAME')
    h_conv5 = tf.nn.relu(conv5 + b_conv5)

    # 5 pooling layer.
    h_pool5 = max_pool_2x2(h_conv5)

    # Fully connected layer 1
    W_fc1 = weight_variable([flat_size, full_connect_num1])
    b_fc1 = bias_variable([full_connect_num1])

    h_pool5_flat = tf.reshape(h_pool5, [-1, flat_size])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat, W_fc1) + b_fc1)

    # Fully connected layer 2
    W_fc2 = weight_variable([full_connect_num1, full_connect_num2])
    b_fc2 = bias_variable([full_connect_num2])
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    # Normalization layer
    phase = tf.placeholder(tf.bool, name='phase')
    h_normal = tf.contrib.layers.batch_norm(h_fc2, center=True, scale=True, is_training=phase)

    # Dropout
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    h_fc2_drop = tf.nn.dropout(h_normal, keep_prob)

    # Map the 1024 features to 2 classes
    W_fc3 = weight_variable([full_connect_num2, classes_num])
    b_fc3 = bias_variable([classes_num])

    y_conv = tf.matmul(h_fc2_drop, W_fc3) + b_fc3
    return y_conv, keep_prob, phase


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)



x = tf.placeholder(tf.float32, [None, img_size_flat], name='x')
y_ = tf.placeholder(tf.float32, [None, classes_num], name='y_')
y_conv, keep_prob, phase = inference(x)
y_conv = tf.identity(y_conv, name="y_conv")

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    # Ensures that we execute the update_ops before performing the train_step
    train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
accuracy = tf.identity(accuracy, name="accuracy")

# read data
batch_size = 50
labels_onehot_dict = [[0, 1], [1, 0]]

img_batch, label_batch = dp.batch('./tfrecords/train.tfrecords', True, img_height, img_width, batch_size)
img_batch_test, label_batch_test = dp.batch('./tfrecords/test.tfrecords', False, img_height, img_width, batch_size)

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator() 
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)    
    
    for i in range(3000):
        batch_x, batch_y = sess.run([img_batch, label_batch])
        batch_x = np.array(batch_x)
        batch_x = batch_x.reshape(batch_size, -1)
        batch_y = np.array([labels_onehot_dict[l] for l in batch_y], dtype=int)

        batch_x_test, batch_y_test = sess.run([img_batch_test, label_batch_test])
        batch_y_test = np.array([labels_onehot_dict[l] for l in batch_y_test], dtype=int)

        train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5, phase: True})
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch_x, y_: batch_y, keep_prob: 1.0, phase: True})
            print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            print('step %d, training accuracy %g' % (i, train_accuracy))
            print('test accuracy %g' % accuracy.eval(feed_dict={
                x: batch_x_test, y_: batch_y_test, keep_prob: 1.0, phase: False}))
            print('train loss %g' % cross_entropy.eval(feed_dict={
                x: batch_x, y_: batch_y, keep_prob: 1.0, phase: True}))
            print('test loss %g' % cross_entropy.eval(feed_dict={
                x: batch_x_test, y_: batch_y_test, keep_prob: 1.0, phase: False}))
        
    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: batch_x_test, y_: batch_y_test, keep_prob: 1.0, phase: False}))
    save_path = saver.save(sess, "./single", global_step=3000)
    print("Model saved in file: %s" % save_path)

    coord.request_stop()
    coord.join(threads)
