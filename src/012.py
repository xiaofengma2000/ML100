import util


def getFile():
    util.downloadTar("https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz",
                     "../data/012/housing.tgz")


getFile()
housing, X, y = util.HousingPrice().data()
print(housing.info())
print(housing["ocean_proximity"].value_counts())
print(housing.describe())

import matplotlib.pyplot as plt
housing.hist(bins=10, figsize=(20,15))
plt.show()
