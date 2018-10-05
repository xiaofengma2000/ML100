from sklearn.datasets import make_moons, fetch_mldata
import tensorflow as tf
from sklearn.model_selection import train_test_split

# m = 1000
# X_moons, y_moons = make_moons(m, noise=0.1, random_state=42)
# y_moons = [y_moons - 0.5]
# X_train, X_test, y_train, y_test = train_test_split(X_moons, y_moons, test_size=0.2, random_state=42)

# feature_cols = [tf.feature_column.numeric_column("X", shape=[X_train.shape[1]])]
# dnn_clf = tf.estimator.DNNClassifier(hidden_units=[300,100], n_classes=2,
#                                      feature_columns=feature_cols)
# input_fn = tf.estimator.inputs.numpy_input_fn(
#     x={"X": X_train}, y=y_train.reshape(-1,1), num_epochs=40, batch_size=50, shuffle=True)
# dnn_clf.train(input_fn=input_fn)
#
# test_input_fn = tf.estimator.inputs.numpy_input_fn(
#     x={"X": X_test}, y=y_test, shuffle=False)
# y_prodect = list(dnn_clf.predict(input_fn=test_input_fn))
# print(y_test)
# print(y_prodect)

m_data = fetch_mldata("MNIST original")
X = m_data["data"]
y = m_data["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 7, random_state=42)

X = tf.placeholder(tf.float32, shape=(None, X_train.shape[1]), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, 200, name="hidden1",
                              activation=tf.nn.elu)
    hidden2 = tf.layers.dense(hidden1, 100, name="hidden2",
                              activation=tf.nn.elu)
    logits = tf.layers.dense(hidden2, 10, name="outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")
    loss_summary = tf.summary.scalar('log_loss', loss)

learning_rate = 0.01
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    sess.run(training_op, feed_dict={X: X_train, y: y_train})

