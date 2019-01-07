from tools.preprocess import Preprocess
from tools.utils import Utilities as uti
import dlib_extractor as dext
from sklearn import svm
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV

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
        self.pp = Preprocess(shuffle=True, compress=False, compress_size=256)
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
    def flatten_features(landmark_points):
        if len(landmark_points.shape) > 3:
            nsamples, nx, ny, rgb = landmark_points.shape
            return landmark_points.reshape((nsamples, nx*ny*rgb))
        else:
            nsamples, nx, ny = landmark_points.shape
            return landmark_points.reshape((nsamples, nx*ny))

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

    def grid_search_best_param(self, estimator, X, y, paramA_type, paramA, paramB_type, paramB):
        parameters = {paramA_type: paramA, paramB_type: paramB}
        grid_obj = GridSearchCV(estimator, parameters, cv=5)
        grid_obj.fit(X, y)
        scores = grid_obj.cv_results_['mean_test_score'].reshape(len(paramA), len(paramB))
        scores = map(list, zip(*scores))

        plt.figure(figsize=(8, 6))
        plt.subplot()
        # plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
        plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot)
        plt.xlabel(paramA_type)
        plt.ylabel(paramB_type)
        plt.colorbar()
        plt.xticks(np.arange(len(paramA)), paramA)
        plt.yticks(np.arange(len(paramB)), paramB)
        plt.title('{} Grid Search AUC Score'.format(self.feature_tested))
        plt.show(block=False)
        plt.savefig('plots/{}_{}_gridsearch_{}.png'.format(paramA_type, paramB_type, self.feature_tested))

    def train_svm(self, training_images, training_labels):

        print "Training ... "

        X = SvmClassification.flatten_features(training_images)
        y = training_labels

        if self.feature_tested == 'hair_color':
            estimator = svm.SVC(gamma='scale', decision_function_shape='ovr', class_weight='balanced', max_iter=2000)
        else:
            estimator = svm.SVC(gamma='scale', decision_function_shape='ovo', class_weight='balanced', max_iter=2000)

        # estimator = svm.SVC(gamma='scale', decision_function_shape='ovr', class_weight='balanced', max_iter=5000)

        # Grid Search to find best parameters
        # kernel = ('linear', 'rbf', 'poly')
        class_weight=['balanced', None]
        decision_function_shape = ['ovr', 'ovo']
        # gamma = ('scale', 'auto')
        # C = [0.001, 0.01, 0.1, 1, 10, 100]
        # degree = [2,3,4]
        print estimator.get_params().keys()
        # self.grid_search_best_param(estimator, X, y, 'decision_function_shape', decision_function_shape, 'class_weight', class_weight)

        # cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
        # print "cv", cv
        # SvmClassification.plot_learning_curve(estimator, "SVM - {}".format(self.feature_tested), X, y, ylim=None, cv=cv, n_jobs=4)
        # plt.savefig('plots/{}_score.png'.format(self.feature_tested))

        estimator.fit(X, training_labels)
        
        return estimator

    def test_svm(self, estimator, test_images, test_labels):

        print "Testing ... "

        new_test_images = SvmClassification.flatten_features(test_images)
        
        y_pred = estimator.predict(new_test_images)

        score = accuracy_score(y_true=test_labels, y_pred=y_pred)

        return score, y_pred


if __name__ == "__main__":
    svm_class = SvmClassification(0.8)

    perf = {}
    for i in range(5):

        # for feature_tested in svm_class.all_classifications:
        for feature_tested in ['smiling']:

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
