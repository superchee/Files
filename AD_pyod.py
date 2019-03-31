import numpy as np
from threading import Thread
import time

from pyod.models.pca import PCA
from pyod.models.ocsvm import OCSVM
from pyod.models.knn import KNN
from pyod.models.abod import ABOD
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.auto_encoder import AutoEncoder

from pyod.utils.utility import *
from sklearn.utils.validation import *
from sklearn.metrics.classification import *
from sklearn.metrics.ranking import *


# a = np.array([1,2,3])
# b = np.array([1,2,3])

# --- get data, split --- #
# -- SWaT data -- #
# data_id = 'swat'
# normal = np.load('./data/normal.npy')
# anomaly = np.load('./data/anomaly.npy')
# # # ALL SENSORS IDX
# # XS = [0, 1, 5, 6, 7, 8, 16, 17, 18, 25, 26, 27, 28, 33, 34, 35, 36, 37, 38, 39, 40, 41, 44, 45, 46, 47]
# # X_n = normal[21600:, XS]
# # X_a = anomaly[:, XS]
# # ALL VARIABLES
# X_train = normal[21600:, 0:51]
# y_train = normal[21600:, 51]
# X_test = anomaly[:, 0:51]
# y_test = anomaly[:, 51]
############################################
# -- KDD99 data -- #
data_id = 'kdd99'
train = np.load('./data/kdd99_train.npy')
test = np.load('./data/kdd99_test.npy')
X_train = train[:, 0:34]
y_train = train[:, 34]
X_test = test[:, 0:34]
y_test = test[:, 34]
############################################
# # -- WADI data -- #
# data_id = 'wadi'
# train = np.load('./data/wadi.npy')
# test = np.load('./data/wadi_a.npy')
#  = train[:, 0:118]
# y_train = train[:, 118]
# X_test = test[:, 0:118]
# y_test = test[:, 118]
############################################
############################################
## -- normalization -- ##
for i in range(X_train.shape[1]):
    # print('i=', i)
    A = max(X_train[:, i])
    if A != 0:
        X_train[:, i] /= max(X_train[:, i])
        X_train[:, i] = 2*X_train[:, i] - 1
    else:
        X_train[:, i] = X_train[:, i]

for i in range(X_test.shape[1]):
    # print('i=', i)
    B = max(X_test[:, i])
    if B != 0:
        X_test[:, i] /= max(X_test[:, i])
        X_test[:, i] = 2*X_test[:, i] - 1
    else:
        X_test[:, i] = X_test[:, i]

X = np.concatenate((X_train, X_test), axis=0)
np.isnan(X)
############################################
#---       evaluation measures         --- #
###########################################
def evaluation_print(clf_name, y, y_pred):
    """
    Utility function for evaluating and printing the results for examples
    Internal use only

    :param clf_name: The name of the detector
    :type clf_name: str

    :param y: The ground truth
    :type y: list or array of shape (n_samples,)

    :param y_pred: The predicted outlier scores
    :type y: list or array of shape (n_samples,)
    """

    # turn raw prediction decision scores into binary labels
    y_pred = get_label_n(y, y_pred)

    # enforce formats of y and labels_
    y = column_or_1d(y)
    y_pred = column_or_1d(y_pred)

    Y_true = y.tolist()
    N = Y_true.count(0)
    P = Y_true.count(1)

    roc = np.round(roc_auc_score(y, y_pred), decimals=4)
    prn = np.round(precision_score(y, y_pred), decimals=4)
    rec = np.round(recall_score(y, y_pred), decimals=4)
    f = np.round((2 * prn * rec / (prn + rec)), decimals=4)
    fp = np.round((P * rec * (1 - prn)) / (prn * N), decimals=4)
    # print('Algorithm:', clf_name)
    # print('Accuracy={}, precision={}, recall={}, f_score={}, false_positive={}'.format(roc,prn,rec,f,fp))
    # return True
    return roc, prn, rec, f, fp

############################################################################################################

#############################################
class myAD_Thread(Thread):
    # the testing data and labels are from above
    def __init__(self, option, data=X_test, label=y_test):
        Thread.__init__(self)
        self.option = option
        self.data = data
        self.label = label
        # self.a = a
        # self.b = b

    def AD_algo(self):

        if self.option == "PCA":
            # --- PCA --- #
            print('testing with PCA...')
            # fit PCA detector
            # clf_name = 'PCA'
            clf_name = self.option
            clf = PCA()
            clf.fit(self.data)

            # get the prediction labels and outlier scores
            y_pca_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
            y_pca_scores = clf.decision_scores_  # raw outlier scores

            # evaluate and print the results
            roc, prn, rec, f1, fp = evaluation_print(clf_name, self.label, y_pca_scores)
            print('%s: Results for Algorithm %s are:' % (self.getName(), clf_name))
            print('Accuracy={}, precision={}, recall={}, f_score={}, false_positive={}'.format(roc, prn, rec, f1, fp))
            # c = self.a + self.b
            # print('result for %s is:' % (self.getName()))
            # print(c)
            f = open("./experiments/plots/AD_pyod.txt", "a")
            f.write('--------------------------------------------\n')
            f.write('%s: Results for Algorithm %s are:\n' % (self.getName(), clf_name))
            f.write('Accuracy={}, precision={}, recall={}, f_score={}, false_positive={}\n'.format(roc, prn, rec, f1, fp))
            f.close()

        elif self.option == "OCSVM":
            # --- OCSVM --- #
            print('testing with OCSVM...')
            # train one_class_svm detector
            # clf_name = 'OneClassSVM'
            clf_name = self.option
            clf = OCSVM()
            clf.fit(self.data)

            # get the prediction labels and outlier scores
            y_ocsvm_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
            y_ocsvm_scores = clf.decision_scores_  # raw outlier scores

            # evaluate and print the results
            roc, prn, rec, f1, fp = evaluation_print(clf_name, self.label, y_ocsvm_scores)
            print('%s: Results for Algorithm %s are:' % (self.getName(), clf_name))
            print('Accuracy={}, precision={}, recall={}, f_score={}, false_positive={}'.format(roc, prn, rec, f1, fp))
            f = open("./experiments/plots/AD_pyod.txt", "a")
            f.write('--------------------------------------------\n')
            f.write('%s: Results for Algorithm %s are:\n' % (self.getName(), clf_name))
            f.write('Accuracy={}, precision={}, recall={}, f_score={}, false_positive={}\n'.format(roc, prn, rec, f1, fp))
            f.close()

        elif self.option == "KNN":
            # --- KNN --- #
            print('testing with KNN...')
            # train kNN detector
            # clf_name = 'KNN'
            clf_name = self.option
            clf = KNN()
            clf.fit(self.data)

            # get the prediction labels and outlier scores
            y_knn_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
            y_knn_scores = clf.decision_scores_  # raw outlier scores

            # evaluate and print the results
            roc, prn, rec, f1, fp = evaluation_print(clf_name, self.label, y_knn_scores)
            print('%s: Results for Algorithm %s are:' % (self.getName(), clf_name))
            print('Accuracy={}, precision={}, recall={}, f_score={}, false_positive={}'.format(roc, prn, rec, f1, fp))
            f = open("./experiments/plots/AD_pyod.txt", "a")
            f.write('--------------------------------------------\n')
            f.write('%s: Results for Algorithm %s are:\n' % (self.getName(), clf_name))
            f.write('Accuracy={}, precision={}, recall={}, f_score={}, false_positive={}\n'.format(roc, prn, rec, f1, fp))
            f.close()

        elif self.option == "ABOD":
            # --- ABOD --- #
            print('testing with ABOD...')
            # train ABOD detector
            # clf_name = 'ABOD'
            clf_name = self.option
            clf = ABOD()
            clf.fit(self.data)

            # get the prediction labels and outlier scores
            y_abod_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
            y_abod_scores = clf.decision_scores_  # raw outlier scores

            # evaluate and print the results
            roc, prn, rec, f1, fp = evaluation_print(clf_name, self.label, y_abod_scores)
            print('%s: Results for Algorithm %s are:' % (self.getName(), clf_name))
            print('Accuracy={}, precision={}, recall={}, f_score={}, false_positive={}'.format(roc, prn, rec, f1, fp))
            f = open("./experiments/plots/AD_pyod.txt", "a")
            f.write('--------------------------------------------\n')
            f.write('%s: Results for Algorithm %s are:\n' % (self.getName(), clf_name))
            f.write('Accuracy={}, precision={}, recall={}, f_score={}, false_positive={}\n'.format(roc, prn, rec, f1, fp))
            f.close()

        elif self.option == "FB":
            # --- FeatureBagging --- #
            print('testing with FeatureBagging...')
            # train FeatureBagging detector
            # clf_name = 'FeatureBagging'
            clf_name = self.option
            clf = FeatureBagging()
            clf.fit(self.data)

            # get the prediction labels and outlier scores
            y_fb_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
            y_fb_scores = clf.decision_scores_  # raw outlier scores

            # evaluate and print the results
            roc, prn, rec, f1, fp = evaluation_print(clf_name, self.label, y_fb_scores)
            print('%s: Results for Algorithm %s are:' % (self.getName(), clf_name))
            print('Accuracy={}, precision={}, recall={}, f_score={}, false_positive={}'.format(roc, prn, rec, f1, fp))
            f = open("./experiments/plots/AD_pyod.txt", "a")
            f.write('--------------------------------------------\n')
            f.write('%s: Results for Algorithm %s are:\n' % (self.getName(), clf_name))
            f.write('Accuracy={}, precision={}, recall={}, f_score={}, false_positive={}\n'.format(roc, prn, rec, f1, fp))
            f.close()

        elif self.option == "AE":
            # --- AutoEncoder --- #
            contamination = 0.1
            print('testing with AutoEncoder...')
            # train AutoEncoder detector
            # clf_name = 'AutoEncoder'
            clf_name = self.option
            clf = AutoEncoder(epochs=30, contamination=contamination)
            clf.fit(self.data)

            # get the prediction labels and outlier scores
            y_ae_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
            y_ae_scores = clf.decision_scores_  # raw outlier scores

            # evaluate and print the results
            roc, prn, rec, f1, fp = evaluation_print(clf_name, self.label, y_ae_scores)
            print('%s: Results for Algorithm %s are:' % (self.getName(), clf_name))
            print('Accuracy={}, precision={}, recall={}, f_score={}, false_positive={}'.format(roc, prn, rec, f1, fp))
            f = open("./experiments/plots/AD_pyod.txt", "a")
            f.write('--------------------------------------------\n')
            f.write('%s: Results for Algorithm %s are:\n' % (self.getName(), clf_name))
            f.write('Accuracy={}, precision={}, recall={}, f_score={}, false_positive={}\n'.format(roc, prn, rec, f1, fp))
            f.close()
#####################################################################


# ###############################################################################################################

if __name__ == "__main__":
    # nameDict = ['PCA', 'OCSVM', 'KNN', 'ABOD', 'FB', 'AE']
    # threads = []
    # nameDict = ['PCA', 'KNN']
    # for idx, algo_name in enumerate(nameDict, 1):
    #     t = threading.Thread(target=myAD_pyod, args=(algo_name,))
    #     threads.append(t)
    #     t.start()
    print('Main Starting...')

    f = open("./experiments/plots/AD_pyod.txt", "a")
    f.write('--------------------------------------------\n')
    f.write('--------------------------------------------\n')
    f.write('Pyod Results for Data-set: %s\n' % data_id)
    f.write('--------------------------------------------\n')
    f.close()

    myThreadOb1 = myAD_Thread("PCA")
    myThreadOb1.setName('Thread 1')

    myThreadOb2 = myAD_Thread("OCSVM")
    myThreadOb2.setName('Thread 2')

    myThreadOb3 = myAD_Thread("KNN")
    myThreadOb3.setName('Thread 3')

    myThreadOb4 = myAD_Thread("ABOD")
    myThreadOb4.setName('Thread 4')

    myThreadOb5 = myAD_Thread("FB")
    myThreadOb5.setName('Thread 5')

    myThreadOb6 = myAD_Thread("AE")
    myThreadOb6.setName('Thread 6')

    # Start running the threads!
    myThreadOb1.AD_algo()
    myThreadOb2.AD_algo()
    myThreadOb3.AD_algo()
    myThreadOb4.AD_algo()
    myThreadOb5.AD_algo()
    myThreadOb6.AD_algo()

    print('Main Terminating...')
