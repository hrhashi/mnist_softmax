from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import argparse
import sys

FLAGS = None

def main(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b

    sess = tf.InteractiveSession()

    # Restore variables from disk.
    saver = tf.train.Saver()
    saver.restore(sess, "/tmp/mnist_softmax_model.ckpt")
    print("Model restored.")

    # Test trained model
    print(sess.run(tf.arg_max(y, 1), feed_dict={x: mnist.test.images[0:10]}))
    print(sess.run(tf.arg_max(mnist.test.labels[0:10], 1)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
