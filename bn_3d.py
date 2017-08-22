import tensorflow as tf
import math
import time
import dataset

data_path = './img'
classes = ['dance', 'nodance']
classes_num = len(classes)
frame_size = 11
img_width = 48
img_height = 72
channel_num = 3
train_ratio = 0.8
video_size_flat = frame_size * img_width * img_height * channel_num

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

flat_size = 1 * 3 * 2 * filter_num5

# Fully connected layer
full_connect_num1 = 4096
full_connect_num2 = 4096

train_data, test_data = dataset.load_data(data_path, classes, frame_size, img_width, img_height, channel_num, train_ratio)

def deepnn(x):

    x_video = tf.reshape(x, [-1, frame_size, img_height, img_width, channel_num])

    # First convolutional layer
    W_conv1 = weight_variable([4, filter_size1, filter_size1, channel_num, filter_num1])
    b_conv1 = bias_variable([filter_num1])
    conv1 = tf.nn.conv3d(x_video, W_conv1, strides=[1, 3, stride1, stride1, 1], padding='SAME')
    h_conv1 = tf.nn.relu(conv1 + b_conv1)

    # # Normalization layer
    # phase = tf.placeholder(tf.bool, name='phase')
    # h_normal1 = tf.contrib.layers.batch_norm(h_conv1, center=True, scale=True, is_training=phase)

    # Pooling layer 
    h_pool1 = max_pool3d(h_conv1)

    # Second convolutional layer
    W_conv2 = weight_variable([2, filter_size2, filter_size2, filter_num1, filter_num2])
    b_conv2 = bias_variable([filter_num2])
    conv2 = tf.nn.conv3d(h_pool1, W_conv2, strides=[1, 2, stride2, stride2, 1], padding='SAME')
    h_conv2 = tf.nn.relu(conv2 + b_conv2)

    # Normalization layer
    # h_normal2 = tf.contrib.layers.batch_norm(h_conv2, center=True, scale=True, is_training=phase)

    # Second pooling layer.
    h_pool2 = max_pool3d(h_conv2)

    # 3 convolutional layer
    W_conv3 = weight_variable([2, filter_size3, filter_size3, filter_num2, filter_num3])
    b_conv3 = bias_variable([filter_num3])
    conv3 = tf.nn.conv3d(h_pool2, W_conv3, strides=[1, 2, stride3, stride3, 1], padding='SAME')
    h_conv3 = tf.nn.relu(conv3 + b_conv3)

    # 4 convolutional layer
    W_conv4 = weight_variable([1, filter_size4, filter_size4, filter_num3, filter_num4])
    b_conv4 = bias_variable([filter_num4])
    conv4 = tf.nn.conv3d(h_conv3, W_conv4, strides=[1, 1, stride4, stride4, 1], padding='SAME')
    h_conv4 = tf.nn.relu(conv4 + b_conv4)

    # 5 convolutional layer
    W_conv5 = weight_variable([1, filter_size5, filter_size5, filter_num4, filter_num5])
    b_conv5 = bias_variable([filter_num5])
    conv5 = tf.nn.conv3d(h_conv4, W_conv5, strides=[1, 1, stride5, stride5, 1], padding='SAME')
    h_conv5 = tf.nn.relu(conv5 + b_conv5)

    # 5 pooling layer.
    h_pool5 = max_pool3d(h_conv5)

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


def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')


def max_pool3d(x):
    return tf.nn.max_pool3d(x, ksize=[1, 1, 2, 2, 1],
                        strides=[1, 1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

x = tf.placeholder(tf.float32, [None, video_size_flat], name='x')
y_ = tf.placeholder(tf.float32, [None, classes_num], name='y_')
y_conv, keep_prob, phase = deepnn(x)
y_conv = tf.identity(y_conv, name="y_conv")

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    # Ensures that we execute the update_ops before performing the train_step
    train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
accuracy = tf.identity(accuracy, name="accuracy")

tf.summary.scalar('cross_entropy', cross_entropy)
tf.summary.scalar('accuracy', accuracy)
saver = tf.train.Saver()

with tf.Session() as sess:

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./bn3d_log' + '/train', sess.graph)
    test_writer = tf.summary.FileWriter('./bn3d_log' + '/test')

    sess.run(tf.global_variables_initializer())
    for i in range(3000):
        train_videos, train_labels = train_data.next_batch(100)
        # train_videos = train_data.videos
        # train_labels = train_data.labels
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: train_videos, y_: train_labels, keep_prob: 1.0, phase: True})
            print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            print('step %d, training accuracy %g' % (i, train_accuracy))
            print('test accuracy %g' % accuracy.eval(feed_dict={
                x: test_data.videos, y_: test_data.labels, keep_prob: 1.0, phase: False}))
            summary, acc = sess.run([merged, accuracy], feed_dict={
                x: test_data.videos, y_: test_data.labels, keep_prob: 1.0, phase: False})
            test_writer.add_summary(summary, i)
            print('train loss %g' % cross_entropy.eval(feed_dict={
                x: train_data.videos, y_: train_data.labels, keep_prob: 1.0, phase: True}))
            print('test loss %g' % cross_entropy.eval(feed_dict={
                x: test_data.videos, y_: test_data.labels, keep_prob: 1.0, phase: False}))
        else:
            # train_step.run(feed_dict={x: train_videos, y_: train_labels, keep_prob: 0.5})
            summary, _ = sess.run([merged, train_step], feed_dict={x: train_videos, y_: train_labels, keep_prob: 0.5, phase: True})
            train_writer.add_summary(summary, i)
    
    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: test_data.videos, y_: test_data.labels, keep_prob: 1.0, phase: False}))
    save_path = saver.save(sess, "./bn_3d", global_step=3000)
    print("Model saved in file: %s" % save_path)
