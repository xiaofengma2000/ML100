import tensorflow as tf
import threading

filenames = tf.placeholder(tf.string, shape=[None])
dataset = tf.contrib.data.CsvDataset(filenames, [[''], [''], [''], [''], [0]], header=True)
dataset = dataset.batch(2)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    print("#1")
    training_filenames = ["../data/006/Social_Network_Ads.csv"]
    sess.run(iterator.initializer, feed_dict={filenames: training_filenames})
    while True:
        try:
            print(sess.run(next_element))
        except tf.errors.OutOfRangeError:
            break
