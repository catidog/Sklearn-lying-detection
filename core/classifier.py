'''
This script includes:

1. ClassifierOfflineTrain
    This is for offline training. The input data are the processed features.

2. class ClassifierOnlineTest(object)
    This is for online testing. The input data are the raw skeletons.
    It uses FeatureGenerator to extract features,
    and then use ClassifierOfflineTrain to recognize the action.
    Notice, this model is only for recognizing the action of one person.

TODO: Add more comments to this function.

'''

import numpy as np
import time
from core.log import setup_logger
logger = setup_logger(name='train model')

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import core.lib_plot as lib_plot
from sklearn.metrics import classification_report

# -- Classes

class ClassifierOfflineTrain(object):
    ''' The classifer for offline training.
        The input features to this classifier are already
            processed by `class FeatureGenerator`.
    '''

    def __init__(self, cfg):
        self.cfg = cfg
        self.classes = self.cfg.classes
        self._init_all_models() # 确定模型类型，初始化分类器参数
        self.clf = self._choose_model(self.cfg.model)

        self.use_pca = self.cfg.pca

    def _choose_model(self, name):
        self.model_name = name
        idx = self.names.index(name)
        return self.classifiers[idx]

    def _init_all_models(self):
        self.names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
                      "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
                      "Naive Bayes", "QDA"]
        self.model_name = None
        self.classifiers = [
            KNeighborsClassifier(5),
            SVC(kernel="linear", C=10.0),
            SVC(gamma=0.01, C=1.0, verbose=True),
            GaussianProcessClassifier(1.0 * RBF(1.0)),
            DecisionTreeClassifier(max_depth=5),
            RandomForestClassifier(
                max_depth=30, n_estimators=100, max_features="auto"),

            MLPClassifier((50, 50, 50)),  # Neural Net

            AdaBoostClassifier(),
            GaussianNB(),
            QuadraticDiscriminantAnalysis()]

    def train(self, X, Y):
        ''' Train model. The result is saved into self.clf '''
        if self.use_pca == True:
            logger.info('Use PCA')
            n_components = min(self.cfg.pca_feature_num, X.shape[1])
            self.pca = PCA(n_components=n_components, whiten=True)
            self.pca.fit(X)

            logger.info("Sum eig values: {}".format(np.sum(self.pca.explained_variance_ratio_)))
            X = self.pca.transform(X)
            logger.info("After PCA, X.shape = {}".format(X.shape))

        logger.info('Start training {} model...'.format(self.model_name))
        self.clf.fit(X, Y)
        logger.info('Training end.')

    def evaluate_model(self, model, tr_X, tr_Y, te_X, te_Y):
        ''' Evaluate accuracy and time cost '''

        # Accuracy
        t0 = time.time()

        tr_accu, tr_Y_predict = model.predict_and_evaluate(tr_X, tr_Y)
        logger.info("Accuracy on training set is {}".format(tr_accu))

        te_accu, te_Y_predict = model.predict_and_evaluate(te_X, te_Y)
        logger.info("Accuracy on testing set is {}".format(te_accu))

        logger.info("Accuracy report:")
        print(classification_report(te_Y, te_Y_predict, target_names=self.classes, output_dict=False))

        # Time cost
        average_time = (time.time() - t0) / (len(tr_Y) + len(te_Y))
        logger.info("Time cost for predicting one sample: "
              "{:.5f} seconds".format(average_time))

        # Plot accuracy
        axis, cf = lib_plot.plot_confusion_matrix(
            te_Y, te_Y_predict, self.classes, normalize=False, size=(12, 8)) # cf 矩阵的具体值
        plt.savefig('output.png', format='png')
        plt.show()

    def predict(self, X):
        ''' Predict the class index of the feature X '''
        if self.use_pca == True:
            X = self.pca.transform(X)
        Y_predict = self.clf.predict(X)
        return Y_predict

    def predict_and_evaluate(self, te_X, te_Y):
        ''' Test model on test set and obtain accuracy '''
        te_Y_predict = self.predict(te_X)
        N = len(te_Y)
        n = sum(te_Y_predict == te_Y)
        accu = n / N
        return accu, te_Y_predict

    def _predict_proba(self, X):
        ''' Predict the probability of feature X belonging to each of the class Y[i] '''
        if self.use_pca == True:
            X = self.pca.transform(X)
        Y_probs = self.clf.predict_proba(X)
        return Y_probs  # np.array with a length of len(classes)





