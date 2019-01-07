from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
from sklearn.utils import class_weight
from keras.applications.mobilenet_v2 import MobileNetV2
from sklearn import metrics

import pandas as pd
import numpy as np
import os

from tools.preprocess import Preprocess
from tools.utils import Utilities as uti
from tools.plotting import Plotting

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

ALL_CLASSIFICATION = ['hair_color', 'eyeglasses', 'smiling', 'young', 'human']

EPOCH_SIZE = 50
INPUT_DIM = 256
BATCH_SIZE = 32
TEST_SPLIT = 0.2


class Cnn:

    def __init__(self, feature_tested, augment, suffix):
        self.feature_tested = feature_tested
        self.augment = augment
        self.suffix = suffix
        if feature_tested == 'hair_color':
            self.multiclass = True
            self.output_dim = 6
            self.class_mode = 'categorical'
        else:
            self.multiclass = False
            self.output_dim = 1
            self.class_mode = 'binary'

        # Define instance variables
        self.pp = None
        self.traindf = None
        self.testdf = None
        self.datagen = None
        self.train_generator = None
        self.valid_generator = None
        self.test_datagen = None
        self.test_generator = None
        self.model = None

    def call_preprocess(self, shuffle, compress, compress_size=None):
        print "Preprocessing Dataset.."
        self.pp = Preprocess(shuffle=shuffle, compress=compress, compress_size=compress_size)
        # noise_free_list = self.pp.filter_noise()
        # train_list, val_list, test_list = self.pp.split_train_val_test(noise_free_list, 1-TEST_SPLIT,0,TEST_SPLIT)
        # self.pp.dir_for_train_val_test(train_list, val_list, test_list)
        # train_path, test_path = self.pp.new_csv(train_list, test_list)
        train_path, test_path = self.pp.new_csv([], [])

        self.traindf = pd.read_csv(train_path, names=['file_name']+ALL_CLASSIFICATION)
        self.testdf = pd.read_csv(test_path, names=['file_name']+ALL_CLASSIFICATION)

        # Filter noise data for hair_color
        if self.feature_tested == 'hair_color':
            print "Removing Noise data ..."
            for frame in [self.traindf, self.testdf]:
                for index, row in frame.iterrows():
                    if row['hair_color'] == -1:
                        frame.drop(index, axis=0, inplace=True)
        else:
            print "Converting for Sigmoid"
            for frame in [self.traindf, self.testdf]:
                for index, row in frame.iterrows():
                    if row[self.feature_tested] == -1:
                        frame[self.feature_tested][index] = 0

    def prepare_generator(self):
        print "Preparing Generators ..."

        if self.augment:
            self.datagen = ImageDataGenerator(rescale=1./255.,
                                              validation_split=0.25,
                                              rotation_range=5,
                                              # width_shift_range=0.2,
                                              # height_shift_range=0.2,
                                              # shear_range=0.1,
                                              zoom_range=0.1,
                                              horizontal_flip=True,
                                              fill_mode='nearest')
        else:
            self.datagen = ImageDataGenerator(rescale=1./255., validation_split=0.25)

        self.train_generator = self.datagen.flow_from_dataframe(dataframe=self.traindf,
                                                                directory=os.path.join(self.pp.dataset_dir, 'training'),
                                                                x_col="file_name",
                                                                y_col=self.feature_tested,
                                                                has_ext=False,
                                                                subset="training",
                                                                batch_size=BATCH_SIZE,
                                                                seed=42,
                                                                shuffle=True,
                                                                class_mode=self.class_mode,
                                                                target_size=(INPUT_DIM, INPUT_DIM))

        self.valid_generator = self.datagen.flow_from_dataframe(dataframe=self.traindf,
                                                                directory=os.path.join(self.pp.dataset_dir, 'training'),
                                                                x_col="file_name",
                                                                y_col=self.feature_tested,
                                                                has_ext=False,
                                                                subset="validation",
                                                                batch_size=BATCH_SIZE,
                                                                seed=42,
                                                                shuffle=True,
                                                                class_mode=self.class_mode,
                                                                target_size=(INPUT_DIM, INPUT_DIM))

        self.test_datagen = ImageDataGenerator(rescale=1./255.)

        self.test_generator = self.test_datagen.flow_from_dataframe(dataframe=self.testdf,
                                                                    directory=os.path.join(self.pp.dataset_dir, 'testing'),
                                                                    x_col="file_name",
                                                                    y_col=self.feature_tested,
                                                                    has_ext=False,
                                                                    batch_size=1,
                                                                    seed=42,
                                                                    shuffle=False,
                                                                    class_mode=self.class_mode,
                                                                    target_size=(INPUT_DIM, INPUT_DIM))

    def custom_test_dataset(self, directory):
        self.test_datagen = ImageDataGenerator(rescale=1. / 255.)
        self.test_generator = self.test_datagen.flow_from_directory(directory=directory,
                                                                    target_size=(256, 256),
                                                                    color_mode="rgb",
                                                                    class_mode=None,
                                                                    batch_size=1,
                                                                    shuffle=False,
                                                                    seed=42,
                                                                    )

    def setup_cnn_model(self, opt):
        print "{} Model setup".format(self.feature_tested)
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), padding='same', input_shape=(INPUT_DIM, INPUT_DIM, 3)))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(32, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(64, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(64, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(256))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))
        
        if opt == 'sgd':
            optimizer = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
        elif opt == 'adam':
            optimizer = optimizers.Adam(lr=0.0001)
        elif opt == 'rmsprop':
            optimizer = optimizers.RMSprop(lr=0.0001, decay=1e-6)
        elif opt == 'adagrad':
            optimizer = optimizers.Adagrad(lr=0.0001, epsilon=None, decay=0.0)
        elif opt == 'adadelta':
            optimizer = optimizers.Adadelta(lr=0.001)
        elif opt == 'adamax':
            optimizer = optimizers.Adamax(lr=0.002)

        if self.multiclass:
            self.model.add(Dense(self.output_dim, activation='softmax'))
            loss = 'categorical_crossentropy'
        else:
            self.model.add(Dense(self.output_dim, activation='sigmoid'))
            loss = 'binary_crossentropy'
            
        self.model.compile(optimizer, loss=loss, metrics=["accuracy"])

    def setup_mlp_model(self, opt):
        self.model = Sequential()
        self.model.add(Flatten(input_shape=(INPUT_DIM, INPUT_DIM, 3)))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.1))

        if opt == 'sgd':
            optimizer = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
        elif opt == 'adam':
            optimizer = optimizers.Adam(lr=0.0001)
        elif opt == 'rmsprop':
            optimizer = optimizers.RMSprop(lr=0.0001, decay=1e-6)
        elif opt == 'adagrad':
            optimizer = optimizers.Adagrad(lr=0.0001, epsilon=None, decay=0.0)
        elif opt == 'adadelta':
            optimizer = optimizers.Adadelta(lr=0.001)
        elif opt == 'adamax':
            optimizer = optimizers.Adamax(lr=0.002)

        if self.multiclass:
            self.model.add(Dense(self.output_dim, activation='softmax'))
            loss = 'categorical_crossentropy'
        else:
            self.model.add(Dense(self.output_dim, activation='sigmoid'))
            loss = 'binary_crossentropy'

        self.model.compile(loss=loss,
                           optimizer=optimizer,
                           metrics=['accuracy'])

    def setup_cnn5_no_fc(self, opt):
        print "{} Model setup".format(self.feature_tested)
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), padding='same', input_shape=(INPUT_DIM, INPUT_DIM, 3)))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(32, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(64, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(64, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(64, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(64, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(256, (8, 8)))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Conv2D(256, (1, 1)))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))

        self.model.add(Conv2D(self.output_dim, (1, 1)))
        self.model.add(Flatten())


        if opt == 'sgd':
            optimizer = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
        elif opt == 'adam':
            optimizer = optimizers.Adam(lr=0.0001)
        elif opt == 'rmsprop':
            optimizer = optimizers.RMSprop(lr=0.0001, decay=1e-6)
        elif opt == 'adagrad':
            optimizer = optimizers.Adagrad(lr=0.0001, epsilon=None, decay=0.0)
        elif opt == 'adadelta':
            optimizer = optimizers.Adadelta(lr=0.001)
        elif opt == 'adamax':
            optimizer = optimizers.Adamax(lr=0.002)

        if self.multiclass:
            self.model.add(Activation('softmax'))
            loss = 'categorical_crossentropy'
        else:
            self.model.add(Activation('sigmoid'))
            loss = 'binary_crossentropy'

        self.model.compile(optimizer, loss=loss, metrics=["accuracy"])

    def setup_mobilenetv2(self):
        self.model = MobileNetV2(input_shape=None, alpha=2.0, depth_multiplier=1, include_top=True, weights=None, input_tensor=None, pooling=None, classes=self.output_dim)
        loss = 'categorical_crossentropy'
        self.model.compile(optimizers.RMSprop(lr=0.0001, decay=1e-6), loss=loss, metrics=["accuracy"])

    def train_model(self):
        print "{} Model fitting...".format(self.feature_tested)
        STEP_SIZE_TRAIN = self.train_generator.n//self.train_generator.batch_size
        STEP_SIZE_VALID = self.valid_generator.n//self.valid_generator.batch_size

        early_stopping = [EarlyStopping(monitor='val_loss',
                                        min_delta=0,
                                        patience=5,
                                        verbose=0,
                                        mode='min',
                                        restore_best_weights=True)]

        truth_list = list(self.traindf[self.feature_tested])
        weights = class_weight.compute_class_weight('balanced', self.train_generator.class_indices.keys(), truth_list)
        weights_dict = {}
        for i in range(len(weights)):
            weights_dict[self.train_generator.class_indices.keys()[i]] = weights[i]

        analysis = self.model.fit_generator(generator=self.train_generator,
                                            steps_per_epoch=STEP_SIZE_TRAIN,
                                            validation_steps=STEP_SIZE_VALID,
                                            validation_data=self.valid_generator,
                                            callbacks=early_stopping,
                                            verbose=1,
                                            workers=4,
                                            class_weight=weights_dict,
                                            epochs=EPOCH_SIZE)
        return analysis

    def evaluate_model(self, generator):
        print "{} Model evaluating...".format(self.feature_tested)
        generator.reset()
        step_size = generator.n//generator.batch_size
        scores = self.model.evaluate_generator(generator=generator, verbose=1, steps=step_size)
        print "Loss: ", scores[0]
        print "Accuracy: ", scores[1]
        return scores

    def saving_model(self):
        # Saving Model
        model_name = "models/{}_{}_{}_".format(self.feature_tested, EPOCH_SIZE, INPUT_DIM)
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(model_name + "model_{}.json".format(self.suffix), "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(model_name + "model_{}.h5".format(self.suffix))
        print "Saved model to disk"

    def predict_model(self, test_generator):
        # Predicts the output of the testing images and returns the dataframe with the output
        test_generator.reset()
        step_size = test_generator.n//test_generator.batch_size
        pred = self.model.predict_generator(test_generator, verbose=2, steps=step_size)
        
        if self.multiclass:
            predicted_class_indices = np.argmax(pred,axis=1)
        else:
            predicted_class_indices = []
            for each in pred:
                if each[0] > 0.5:
                    predicted_class_indices.append(1)
                else:
                    predicted_class_indices.append(0)

        labels = self.train_generator.class_indices
        labels = dict((v, k) for k, v in labels.items())
        predictions = [labels[k] for k in predicted_class_indices]

        results = uti.to_dataframe(test_generator.filenames, predictions)

        return results

    def manual_check_model(self, results):
        truth_list = list(self.testdf[self.feature_tested])
        predictions = list(results['Predictions'])
        average = 'binary'
        if self.multiclass:
            average = 'weighted'
        f1_score = metrics.f1_score(truth_list, predictions, average=average)

        count = 0
        for i in range(len(truth_list)):
            if truth_list[i] == predictions[i]:
                count += 1
        accuracy = count/float(len(truth_list))
        print "Accuracy: {:.2f}%".format(accuracy*100)
        return accuracy, f1_score

    def save_csv(self, dataframe, accuracy):
        if self.feature_tested != 'hair_color':
            print "Zeroes back to -1 for binary cases"
            for index, row in dataframe.iterrows():
                if row['Predictions'] == 0:
                    dataframe['Predictions'][index] = -1

        # Saving Results into csv
        print dataframe.head()
        csv_file = "results/{}_{}_{}_results_{}.csv".format(self.feature_tested, EPOCH_SIZE, INPUT_DIM, self.suffix)
        dataframe.to_csv(csv_file,index=False)

        data = open(csv_file, 'r').readlines()[1:]
        data.insert(0, '{},,\n'.format(accuracy))
        open(csv_file, 'w').writelines(data)


if __name__ == "__main__":

    for feature in ALL_CLASSIFICATION:
        cnn = Cnn(feature, augment=True, suffix='cnn3+fc-FINAL')
        cnn.call_preprocess(shuffle=False, compress=False, compress_size=INPUT_DIM)
        cnn.prepare_generator()

        cnn.custom_test_dataset(os.path.join(cnn.pp.dataset_dir, "..", "testing_prediction","dataset"))

        # cnn.setup_mlp_model('sgd')
        cnn.setup_cnn_model('adam')
        # cnn.setup_cnn5_no_fc('adam')
        # cnn.setup_mobilenetv2()
        cnn.model.summary()

        history = cnn.train_model()
        Plotting.plot_history(history, '', EPOCH_SIZE, cnn.feature_tested, cnn.suffix, save=True, show=False)
        cnn.evaluate_model(cnn.valid_generator)
        # # results = cnn.evaluate_model(cnn.test_generator)
        cnn.saving_model()

        dataframe = cnn.predict_model(cnn.test_generator)
        # accuracy, f1 = cnn.manual_check_model(dataframe)
        # print 'f1 score:', f1
        cnn.save_csv(dataframe, 0)
