import tensorflow as tf
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
import numpy as np

mnist = fetch_mldata('MNIST original')
X_train, X_test, y_train, y_test = train_test_split(mnist["data"], mnist["target"], test_size=0.2, random_state=10)
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]


with tf.name_scope("place"):
    X = tf.placeholder(tf.float32, shape=(None, X_train.shape[1]), name='X')
    y = tf.placeholder(tf.int32, shape=(None), name='y')

with tf.name_scope("DNN"):
    hidden1 = tf.layers.dense(X, 300, activation=tf.nn.relu, name="hidden1")
    drop1 = tf.layers.dropout(hidden1, rate=0.8, name="drop1")
    hidden2 = tf.layers.dense(drop1, 100, activation=tf.nn.relu, name="hidden2")
    logits = tf.layers.dense(hidden2, 10, name="outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")
    # loss = tf.losses.mean_squared_error(labels=y, predictions=logits)

with tf.name_scope("train"):
    optimizer = tf.train.MomentumOptimizer(0.01, 0.9)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

n_epochs = 40
batch_size = 50

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        if epoch % 10 == 0:
            acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_valid = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
            print(epoch, "Batch accuracy:", acc_batch, "Validation accuracy:", acc_valid)
    acc_final = accuracy.eval(feed_dict={X: X_test, y: y_test})
    print("Final accuracy:", acc_final)
