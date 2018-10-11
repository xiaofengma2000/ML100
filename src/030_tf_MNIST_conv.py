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
    training = tf.placeholder(tf.bool, shape=(None), name='training')

with tf.name_scope("CNN"):
    # hidden1 = tf.layers.dense(X, 300, activation=tf.nn.relu, name="hidden1")
    X_reshaped = tf.reshape(X, shape=[-1, 28, 28, 1])
    conv1 = tf.layers.conv2d(X_reshaped, filters=32, kernel_size=[5,5],
                         strides=1, padding="SAME",
                         activation=tf.nn.relu, name="conv1")
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    pool_flat = tf.reshape(pool1, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=training) #tf.estimator.ModeKeys.TRAIN)
    logits = tf.layers.dense(inputs=dropout, units=10)

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
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch, training: True})
        if epoch % 10 == 0:
            acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch, training : False })
            acc_valid = accuracy.eval(feed_dict={X: X_valid, y: y_valid, training : False})
            print(epoch, "Batch accuracy:", acc_batch, "Validation accuracy:", acc_valid)
    acc_final = accuracy.eval(feed_dict={X: X_test, y: y_test, training : False})
    print("Final accuracy:", acc_final)
