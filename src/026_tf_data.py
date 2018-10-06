import tensorflow as tf
import threading

def read_my_file_format(filename_queue):
    def decode_csv(line):
        parsed_line = tf.decode_csv(line, record_defaults=[[''], [''], [''], [''], ['']])
        return parsed_line

    ds = tf.data.TextLineDataset(filename_queue).skip(1)
    return ds.map(decode_csv)


filenames = tf.placeholder(tf.string, shape=[None])
dataset = tf.data.TFRecordDataset(filenames)
dataset.flat_map(read_my_file_format)

# dataset = tf.contrib.data.CsvDataset(filenames, [[''], [''], [''], [''], [0]], header=True)
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
