import tensorflow as tf
from housing import HousingPrice

X_train, X_test, y_train, y_test = HousingPrice().prepare_data()

X = tf.placeholder(shape=(None, X_train.shape[1]), dtype=tf.float32, name="X")
y = tf.placeholder(shape=(None, 1), dtype=tf.float32, name="y")

theta = tf.Variable(tf.random_uniform([X_train.shape[1], 1], -1.0, 1.0, seed=42), dtype=tf.float32, name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
init = tf.global_variables_initializer()
training_op = optimizer.minimize(mse)
with tf.Session() as sess:
    tf.train.Saver().restore(sess, "../model/024.ckpt")
    print(theta.eval())




