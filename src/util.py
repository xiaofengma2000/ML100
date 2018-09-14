import pandas
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class Tennis:

    def first_set_result(self, row):
        if row['ST1.1'] < row['ST1.2']:
            return 0
        return 1

    def data(self):
        rawdata = pandas.read_csv("../data/tennis/AusOpen-men-2013.csv")
        # rawdata=pandas.read_csv("../data/tennis/AusOpen-women-2013.csv")

        # data = rawdata[['Player1','Player2','Result','FSP.1','FSW.1','SSP.1','SSW.1','FSP.2','FSW.2','SSP.2','SSW.2']]
        rawdata['ST1_Result'] = rawdata.apply(lambda row: self.first_set_result(row), axis=1)
        data = rawdata[['Result', 'FSW.1', 'SSW.1', 'FSW.2', 'SSW.2', 'ST1_Result']]
        print(data)

        x = data.iloc[:, 1:].values
        y = data.iloc[:, 0].values

        onehotencoder = OneHotEncoder(categorical_features=[4])
        x = onehotencoder.fit_transform(x).toarray()

        return x,y

class SocialAd:

    def data(self):
        data = pandas.read_csv("../data/006/Social_Network_Ads.csv")

        x = data.iloc[:, 1:4].values
        y = data.iloc[:, -1].values

        x[:, 0] = LabelEncoder().fit_transform(x[:, 0])

        return x,y

class SmsSpam:

    def data(self):
        # rawdata = pandas.read_csv("../data/smsspam/SMSSpamCollection", delimiter="	", header=None)
        rawdata = pandas.read_table("../data/smsspam/SMSSpamCollection", header=None)
        x = rawdata.iloc[:, 1].values
        y = rawdata.iloc[:, 0].values

        # y = LabelEncoder().fit_transform(y)
        return x,y

class Report:

    def ROC(self,x_test, y_test, classifier):
        logit_roc_auc = roc_auc_score(y_test, classifier.predict(x_test))
        fpr, tpr, _ = roc_curve(y_test, classifier.predict_proba(x_test)[:, 1])
        plt.figure()
        plt.plot(fpr, tpr, label='KNN (area = %0.2f)' % logit_roc_auc)
        plt.plot([0, 1], [0, 1], 'g--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # plt.title('Tennis characteristic')
        plt.legend(loc="lower right")
        # plt.savefig('Log_ROC')
        plt.show()

    def PCA(self, x, y):
        pca = PCA(n_components=2).fit_transform(x)
        plt.figure(dpi=120)
        plt.scatter(pca[y == 0, 0], pca[y == 0, 1], alpha=0.5, label='YES', s=2, color='navy')
        plt.scatter(pca[y == 1, 0], pca[y == 1, 1], alpha=0.5, label='NO', s=2, color='darkorange')
        plt.legend()
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.gca().set_aspect('equal')
        plt.show()
