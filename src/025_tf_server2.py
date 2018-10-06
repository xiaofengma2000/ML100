import threading
import tensorflow as tf

cluster_spec = tf.train.ClusterSpec({
    "ps": [
        "127.0.0.1:2221",  # /job:ps/task:0
        "127.0.0.1:2222",  # /job:ps/task:1
    ],
    "worker": [
        "127.0.0.1:2223",  # /job:worker/task:0
        "127.0.0.1:2224",  # /job:worker/task:1
        "127.0.0.1:2225",  # /job:worker/task:2
    ]})

def startlocalservers(cluster_spec):
    task_ps0 = tf.train.Server(cluster_spec, job_name="ps", task_index=0)
    task_ps1 = tf.train.Server(cluster_spec, job_name="ps", task_index=1)
    task_worker0 = tf.train.Server(cluster_spec, job_name="worker", task_index=0)
    task_worker1 = tf.train.Server(cluster_spec, job_name="worker", task_index=1)
    task_worker2 = tf.train.Server(cluster_spec, job_name="worker", task_index=2)

# threading.Thread(target=startlocalservers, args=[cluster_spec]).start()
startlocalservers(cluster_spec)

with tf.device("/job:ps"):
    a = tf.Variable(1.0, name="a")

with tf.device("/job:worker"):
    b = a + 2

with tf.device("/job:worker/task:1"):
    c = a + b

config = tf.ConfigProto()
config.log_device_placement = True

print("start training...")
with tf.Session("grpc://127.0.0.1:2221", config=config) as sess:
    sess.run(a.initializer)
    print(c.eval())
