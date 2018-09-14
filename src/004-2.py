import numpy,pandas
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


def first_set_result (row):
   if row['ST1.1'] < row['ST1.2'] :
      return 0
   return 1

rawdata=pandas.read_csv("../data/tennis/AusOpen-men-2013.csv")
# rawdata=pandas.read_csv("../data/tennis/AusOpen-women-2013.csv")

# data = rawdata[['Player1','Player2','Result','FSP.1','FSW.1','SSP.1','SSW.1','FSP.2','FSW.2','SSP.2','SSW.2']]
rawdata['ST1_Result'] = rawdata.apply (lambda row: first_set_result (row),axis=1)
data = rawdata[['Result','FSW.1','SSW.1','FSW.2','SSW.2','ST1_Result']]
print(data)

# sns.countplot(x="SSW.2", data=data, palette='hls')
# plt.show()

# print(data.groupby('Result').mean())

# 1st set result vs final result
# pandas.crosstab(data['ST1_Result'],data['Result']).plot(kind='bar')
# plt.show()


x = data.iloc[ :, 1:]
y = data.iloc[ :, 0]

onehotencoder = OneHotEncoder(categorical_features = [4])
x = onehotencoder.fit_transform(x).toarray()

from sklearn.decomposition import PCA
pca = PCA(n_components=2).fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(pca, y, random_state=0)

print(pca)
plt.figure(dpi=120)
plt.scatter(pca[y.values==0,0], pca[y.values==0,1], alpha=0.5, label='YES', s=2, color='navy')
plt.scatter(pca[y.values==1,0], pca[y.values==1,1], alpha=0.5, label='NO', s=2, color='darkorange')
plt.legend()
plt.title('Bank Marketing Data Set\nFirst Two Principal Components')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.gca().set_aspect('equal')
plt.show()

