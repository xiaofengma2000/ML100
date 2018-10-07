import tensorflow as tf
import pandas

csv = pandas.read_csv("../data/006/Social_Network_Ads.csv").values
with tf.python_io.TFRecordWriter("../data/006/Social_Network_Ads.tfrecords") as writer:
    for row in csv:
        # features, label = row[:-1], row[-1]
        # Gender,Age,EstimatedSalary,Purchased
        example = tf.train.Example()
        example.features.feature["User ID"].int64_list.value.append(row[0])
        # example.features.feature["Gender"].bytes_list.value.append(row[1])
        example.features.feature["Age"].int64_list.value.append(row[2])
        example.features.feature["EstimatedSalary"].int64_list.value.append(row[3])
        example.features.feature["Purchased"].int64_list.value.append(row[4])
        writer.write(example.SerializeToString())
