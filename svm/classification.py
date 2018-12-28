from preprocess import Preprocess
import dlib_extractor as dext
from sklearn import svm
import numpy as np
import random
import os

NPY_FILE_DIR = 'features_and_labels/'
feature = 'face_features'
# feature = 'ds_images'

class SvmClassification():
    def __init__(self, train_split):
        self.pp = Preprocess()
        self.data_list = self.pp.filter_noise()
        self.train_split = train_split
        self.all_classifictions = ['hair_color', 'eyeglasses', 'smiling', 'young','human']

    def get_data(self, feature_tested):

        if not os.path.exists(NPY_FILE_DIR+feature+'.npy') or not os.path.exists(NPY_FILE_DIR+feature_tested + '_labels.npy'):
            dext.extract_features_labels(self.data_list)

        X, y = dext.load_features_extract_labels(NPY_FILE_DIR+feature+'.npy', NPY_FILE_DIR+feature_tested + '_labels.npy')

        train_list = random.sample(range(len(X)),int(self.train_split*len(X)))
        train_list.sort()
        trainname_list = [self.data_list[x] for x in train_list]
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


    def train_SVM(self, training_images, training_labels):
        print "Training ... "

        if len(training_images.shape)>3:
            nsamples, nx, ny, rgb = training_images.shape
            reshaped_training_images = training_images.reshape((nsamples,nx*ny*rgb))
        else:
            nsamples, nx, ny = training_images.shape
            reshaped_training_images = training_images.reshape((nsamples,nx*ny))

        
        clf = svm.SVC(gamma='scale')
        clf.fit(reshaped_training_images, training_labels)
        
        return clf

    def test_SVM(self, clf, test_images, test_labels):
        print "Testing ... "

        if len(test_images.shape)>3:
            nsamples, nx, ny, rgb = test_images.shape
            new_test_images = test_images.reshape((nsamples,nx*ny*rgb))
        else:
            nsamples, nx, ny = test_images.shape
            new_test_images = test_images.reshape((nsamples,nx*ny))
        
        arr = clf.predict(new_test_images)
        count = 0
        for i in range(len(test_images)):
            if arr[i]==test_labels[i]:
                count+=1
        return (float(count)/len(test_images)), arr


if __name__ == "__main__":
    svm_class = SvmClassification(0.75)

    perf = {}
    for i in range(1):
        for feature_tested in svm_class.all_classifictions:
            tr_data,tr_lbl,te_data,te_lbl = svm_class.get_data(feature_tested)
            clf = svm_class.train_SVM(tr_data,tr_lbl)
            accuracy, arr = svm_class.test_SVM(clf, te_data, te_lbl)
            print "SVM Accuracy '{}': {:.2f}%".format(feature_tested,accuracy*100)
            if feature_tested not in perf:
                perf[feature_tested] = []
            perf[feature_tested].append(accuracy)
            svm_class.pp.save_csv(svm_class.testname_list, arr, accuracy, 'results/'+feature_tested+"_svm.csv")

    for key in perf:
        print key, sum(perf[key])/float(len(perf[key]))
