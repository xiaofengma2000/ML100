import tensorflow as tf
from housing import HousingPrice

X_train, X_test, y_train, y_test = HousingPrice().prepare_data()

with tf.name_scope("place"):
    X = tf.placeholder(tf.float32, shape=(None, X_train.shape[1]), name='X')
    y = tf.placeholder(tf.float32, shape=(None), name='y')

with tf.name_scope("DNN"):
    hidden1 = tf.layers.dense(X, 100, activation=tf.nn.relu, name="hidden1")
    drop1 = tf.layers.dropout(hidden1, rate=0.8, name="drop1")
    hidden2 = tf.layers.dense(drop1, 80, activation=tf.nn.relu, name="hidden2")
    logits = tf.layers.dense(hidden2, 1, name="outputs")

with tf.name_scope("loss"):
    # predictions = tf.squeeze(logits, 1)
    loss = tf.losses.mean_squared_error(labels=y, predictions=logits)

with tf.name_scope("train"):
    optimizer = tf.train.MomentumOptimizer(0.01, 0.9)
    training_op = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2):
        sess.run(training_op, feed_dict={X: X_train, y: y_train})
        loss_val = loss.eval(feed_dict={X: X_train, y: y_train})
        print('epoch #', i)
        print(loss_val)
        loss_val = loss.eval(feed_dict={X: X_test, y: y_test})
        print(loss_val)
