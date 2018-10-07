import tensorflow as tf
import threading

COLUMNS = ['User ID', 'Gender',
           'Age', 'EstimatedSalary',
           'Purchased']


def read_my_file_format(filename_queue):
    ds = tf.data.TextLineDataset(filename_queue).skip(1)

    def decode_csv(line):
        fields = tf.decode_csv(line, record_defaults=[[0], [''], [0], [0], [0]])
        features = dict(zip(COLUMNS, fields))
        label = features.pop('Purchased')
        # print(features)
        return features, label

    ds = ds.map(decode_csv).batch(20)
    iterator = ds.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels


next_batch = read_my_file_format('../data/006/Social_Network_Ads.csv')

with tf.Session() as sess:
    while True:
        try:
            print(sess.run(next_batch))
        except tf.errors.OutOfRangeError:
            print('EOF')
            break
