import tensorflow as tf
import threading

# Gender,Age,EstimatedSalary,Purchased
def _extract_features(example):
    features = {
        "User ID": tf.VarLenFeature(tf.int64),
        # "Gender": tf.VarLenFeature(tf.string),
        "Age": tf.VarLenFeature(tf.int64),
        "EstimatedSalary": tf.VarLenFeature(tf.int64),
        "Purchased": tf.VarLenFeature(tf.int64)
    }
    parsed_example = tf.parse_single_example(example, features)
    return tf.sparse_tensor_to_dense(parsed_example["User ID"]), tf.sparse_tensor_to_dense(parsed_example["Purchased"])


dataset = tf.data.TFRecordDataset('../data/006/Social_Network_Ads.tfrecords')
dataset = dataset.map(_extract_features)
dataset = dataset.batch(2)
# dataset = dataset.shuffle(buffer_size=50)
# dataset = dataset.repeat(1)
iterator = dataset.make_one_shot_iterator()
i, j = iterator.get_next()

with tf.Session() as sess:
    while True:
        try:
            print(sess.run([i, j]))
        except tf.errors.OutOfRangeError:
            break
