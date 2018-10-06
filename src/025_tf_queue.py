import tensorflow as tf
import threading

filename = tf.placeholder(tf.string)
filename_queue = tf.FIFOQueue(capacity=10, dtypes=[tf.string], shapes=[()])
enqueue_filename = filename_queue.enqueue([filename])
close_filename_queue = filename_queue.close()

reader = tf.TextLineReader(skip_header_lines=1)
key, value = reader.read(filename_queue)
x1, x2, x3, x4, target = tf.decode_csv(value, record_defaults=[[''],[''],[''],[''],[0]])
features = tf.stack([x1, x2, x3, x4])

content_queue = tf.RandomShuffleQueue(capacity=10, min_after_dequeue=2, dtypes=[tf.string, tf.int32], shapes=[[4], []])
enqueue_instance = content_queue.enqueue([features, target])
close_instance_queue = content_queue.close()

minibatch_instances, minibatch_targets = content_queue.dequeue_up_to(2)


def enque_then_close(session, tensor, tensor_close):
    try:
        while True:
            session.run(tensor)
    except tf.errors.OutOfRangeError as ex:
        print("No more files to read")
    session.run(tensor_close)


with tf.Session() as sess:
    print("#1")
    sess.run(enqueue_filename, feed_dict={filename: "../data/006/Social_Network_Ads.csv"})
    print("#2")
    sess.run(close_filename_queue)
    # print("#3")
    # try:
    #     while True:
    #         sess.run(enqueue_instance)
    # except tf.errors.OutOfRangeError as ex:
    #     print("No more files to read")
    # print("#4")
    # sess.run(close_instance_queue)

    threading.Thread(target=enque_then_close, args=[sess, enqueue_instance, close_instance_queue]).start()

    print("#5")
    try:
        while True:
            print(sess.run([minibatch_instances, minibatch_targets]))
    except tf.errors.OutOfRangeError as ex:
        print("No more training instances")


