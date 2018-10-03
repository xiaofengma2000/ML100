import tensorflow as tf

x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x*x + x*y + y*y + 7

init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    result = f.eval()

print(result)
