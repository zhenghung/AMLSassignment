from tools.preprocess import Preprocess
from tools.utils import Utilities as uti
import dlib_extractor as dext
from sklearn import svm
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import validation_curve
from sklearn.linear_model import Ridge

import numpy as np
import random
import os
import matplotlib.pyplot as plt

NPY_FILE_DIR = 'features_and_labels/'
feature = 'face_features'
# feature = 'ds_images'
# feature = 'edges'


class SvmClassification:
    def __init__(self, train_split):
        print "Sampling Training and Testing data ..."
        self.pp = Preprocess(shuffle=True, compress=False)
        self.data_list = self.pp.filter_noise()
        self.train_split = train_split
        self.all_classifications = ['hair_color', 'eyeglasses', 'smiling', 'young', 'human']
        self.testname_list = None
        self.feature_tested = None

    def get_data(self, feature_tested):
        self.feature_tested = feature_tested
        if not os.path.exists(NPY_FILE_DIR+feature+'.npy') or not os.path.exists(NPY_FILE_DIR+feature_tested + '_labels.npy'):
            dext.extract_features_labels(self.data_list)

        X, y = dext.load_features_extract_labels(NPY_FILE_DIR+feature+'.npy', NPY_FILE_DIR+feature_tested + '_labels.npy')

        train_list = random.sample(range(len(X)), int(self.train_split*len(X)))
        train_list.sort()

        test_list = []
        for i in range(len(X)):
            if i not in train_list:
                test_list.append(i)

        self.testname_list = [self.data_list[x]+'.png' for x in test_list]

        tr_X = np.array([X[x] for x in train_list])
        tr_Y = np.array([y[x] for x in train_list])
        te_X = np.array([X[x] for x in test_list])
        te_Y = np.array([y[x] for x in test_list])
        return tr_X, tr_Y, te_X, te_Y

    @staticmethod
    def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10)):

        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")

        plt.legend(loc="best")
        return plt

    def train_svm(self, training_images, training_labels):

        print "Training ... "

        if len(training_images.shape)>3:
            nsamples, nx, ny, rgb = training_images.shape
            reshaped_training_images = training_images.reshape((nsamples, nx*ny*rgb))
        else:
            nsamples, nx, ny = training_images.shape
            reshaped_training_images = training_images.reshape((nsamples, nx*ny))

        X = reshaped_training_images
        y = training_labels

        estimator = svm.SVC(gamma='scale', decision_function_shape='ovr', class_weight='balanced')

        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
        SvmClassification.plot_learning_curve(estimator, "SVM - {}".format(self.feature_tested), X, y, ylim=None, cv=cv, n_jobs=4)
        plt.show(block=False)

        estimator.fit(reshaped_training_images, training_labels)
        
        return estimator

    def test_svm(self, estimator, test_images, test_labels):

        print "Testing ... "

        if len(test_images.shape) > 3:
            nsamples, nx, ny, rgb = test_images.shape
            new_test_images = test_images.reshape((nsamples, nx*ny*rgb))
        else:
            nsamples, nx, ny = test_images.shape
            new_test_images = test_images.reshape((nsamples, nx*ny))
        
        arr = estimator.predict(new_test_images)
        count = 0
        for i in range(len(test_images)):
            if arr[i] == test_labels[i]:
                count += 1
        return (float(count)/len(test_images)), arr


if __name__ == "__main__":
    svm_class = SvmClassification(0.8)

    perf = {}
    for i in range(1):

        for feature_tested in svm_class.all_classifications:
        # for feature_tested in ['hair_color']:

            tr_data, tr_lbl, te_data, te_lbl = svm_class.get_data(feature_tested)
            clf = svm_class.train_svm(tr_data, tr_lbl)
            accuracy, pred_list = svm_class.test_svm(clf, te_data, te_lbl)
            print "SVM Accuracy '{}': {:.2f}%".format(feature_tested, accuracy*100)

            if feature_tested not in perf:
                perf[feature_tested] = []
            perf[feature_tested].append(accuracy)

            df = uti.to_dataframe(svm_class.testname_list, pred_list)
            uti.save_csv(df, accuracy, feature_tested+"_svm")

    print "====================="
    for key in perf:
        print key, sum(perf[key])/float(len(perf[key]))
    plt.show()