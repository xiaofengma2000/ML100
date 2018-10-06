import threading
import tensorflow as tf

def startlocalserver():
    server = tf.train.Server.create_local_server()
    with tf.Session(server.target) as sess:
        print(sess.run(tf.constant("Hello distributed TensorFlow!")))

threading.Thread(target=startlocalserver()).start()

