import tensorflow as tf
import numpy as np
from PIL import Image

def is_dance(file_path):
    image = Image.open(file_path)
    image = np.array(image.resize((48, 72)))
    images = np.array([image])
    images = images.astype(np.float32)
    images = np.multiply(images, 1.0 / 255.0)
    images = images.reshape(1, -1)
 
    sess=tf.Session()    

    saver = tf.train.import_meta_graph('single2-3000.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./'))

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")
    y_conv = graph.get_tensor_by_name("y_conv:0")
    feed_dict ={x: images, keep_prob: 1.0}

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax([[1, 0]], 1))
    print(sess.run(correct_prediction, feed_dict))
    print(sess.run(y_conv, feed_dict))


is_dance('WX20170726-191836.jpg')