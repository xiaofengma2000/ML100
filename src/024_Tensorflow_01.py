import tensorflow as tf
from util import SocialAd
from sklearn.model_selection import train_test_split
from housing import HousingPrice

X_train, X_test, y_train, y_test = HousingPrice().prepare_data()
print(X_train)

# X = tf.placeholder(shape=(None, X_train.shape[1]), dtype=tf.float32, name="X")
# y = tf.placeholder(shape=(None, 1), dtype=tf.float32, name="y")

X = tf.constant(X_train, dtype=tf.float32, name="X")
y = tf.constant(y_train.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([X_train.shape[1], 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
n_epochs = 1000
init = tf.global_variables_initializer()
training_op = optimizer.minimize(mse)
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
        sess.run(training_op)
        best_theta = theta.eval()

print("Best theta:")
print(best_theta)

