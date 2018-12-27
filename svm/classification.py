from preprocess import Preprocess
import dlib_extractor as dext
import numpy as np
from sklearn import svm
import tensorflow as tf
import time
import random

pp = Preprocess()
data_list = pp.filter_noise()
test_split = 0.2

def get_data():
    global train_list
    global test_list
    X, y = dext.extract_features_labels(data_list, 3)

    train_list = random.sample(range(len(X)),int(test_split*len(X)))
    train_list.sort()
    trainname_list = [data_list[x] for x in train_list]
    print trainname
    return

    test_list = []
    ptr = 0
    for i in range(len(X)):
        # print ptr
        if i not in train_list:
            test_list.append(i)

    testname_list = [data_list[x] for x in test_list]

    tr_X = np.array([X[x] for x in train_list])
    tr_Y = np.array([y[x] for x in train_list])
    te_X = np.array([X[x] for x in test_list])
    te_Y = np.array([y[x] for x in test_list])
    return tr_X, tr_Y, te_X, te_Y


def train_SVM(training_images, training_labels):
    # X, y = dext.extract_features_labels(data_list)
    # training_labels = np.array([y[x] for x in train_list])
    # test_labels = np.array([y[x] for x in test_list])
    nsamples, nx, ny = training_images.shape
    reshaped_training_images = training_images.reshape((nsamples,nx*ny))
    
    clf = svm.SVC(gamma='scale')
    clf.fit(reshaped_training_images, training_labels)
    
    return clf

def test_SVM(clf, test_images, test_labels):

    nsamples, nx, ny = test_images.shape
    new_test_images = test_images.reshape((nsamples,nx*ny))
    arr = clf.predict(new_test_images)
    count = 0
    for i in range(len(test_images)):
        if arr[i]==test_labels[i]:
            count+=1
    return (float(count)/len(test_images))*100, arr


def train_MLP(training_images, training_labels, test_images, test_labels):


    return 0

a,b,c,d = get_data()


# SVM
clf = train_SVM(a,b)
num, arr = test_SVM(clf, c, d)
print "SVM Accuracy: {:.2f}%".format(num)
pp.save_csv(test_list, arr, num, "svm.csv")


# # MLP
# num2 = train_MLP(a,b,c,d)
# print "MLP Accuracy: {:.2f}%".format(num2)

