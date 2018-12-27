from preprocess import Preprocess
import dlib_extractor as dext
from sklearn import svm
import numpy as np
import random
import os

NPY_FILE_DIR = 'features_and_labels/'

pp = Preprocess()
data_list = pp.filter_noise()
train_split = 0.8
all_classifictions = ['hair_color', 'eyeglasses', 'smiling', 'young','human']

def get_data(feature_tested):
    global train_list
    global testname_list
    global truth_list

    if not os.path.exists(NPY_FILE_DIR+'features.npy') or not os.path.exists(NPY_FILE_DIR+feature_tested + '_labels.npy'):
        dext.extract_features_labels(data_list)

    X, y = dext.load_features_extract_labels(NPY_FILE_DIR+'features.npy', NPY_FILE_DIR+feature_tested + '_labels.npy')

    train_list = random.sample(range(len(X)),int(train_split*len(X)))
    train_list.sort()
    trainname_list = [data_list[x] for x in train_list]
    test_list = []
    for i in range(len(X)):
        if i not in train_list:
            test_list.append(i)

    testname_list = [data_list[x] for x in test_list]
    truth_list = [y[x] for x in test_list]

    tr_X = np.array([X[x] for x in train_list])
    tr_Y = np.array([y[x] for x in train_list])
    te_X = np.array([X[x] for x in test_list])
    te_Y = np.array([y[x] for x in test_list])
    return tr_X, tr_Y, te_X, te_Y


def train_SVM(training_images, training_labels):
    print "Training ... "
    nsamples, nx, ny = training_images.shape
    reshaped_training_images = training_images.reshape((nsamples,nx*ny))
    
    clf = svm.SVC(gamma='scale')
    clf.fit(reshaped_training_images, training_labels)
    
    return clf

def test_SVM(clf, test_images, test_labels):
    print "Testing ... "
    nsamples, nx, ny = test_images.shape
    new_test_images = test_images.reshape((nsamples,nx*ny))
    arr = clf.predict(new_test_images)
    count = 0
    for i in range(len(test_images)):
        if arr[i]==test_labels[i]:
            count+=1
    return (float(count)/len(test_images))*100, arr

for feature_tested in all_classifictions:

    tr_data,tr_lbl,te_data,te_lbl = get_data(feature_tested)

    # SVM
    clf = train_SVM(tr_data,tr_lbl)
    accuracy, arr = test_SVM(clf, te_data, te_lbl)
    print "SVM Accuracy '{}': {:.2f}%".format(feature_tested,accuracy)
    pp.save_csv(testname_list, arr, accuracy, 'results/'+feature_tested+"_svm.csv")


