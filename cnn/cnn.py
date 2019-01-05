from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
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

    def call_preprocess(self, shuffle, compress):
        print "Preprocessing Dataset.."
        self.pp = Preprocess(shuffle=shuffle, compress=compress)
        noise_free_list = self.pp.filter_noise()
        train_list, val_list, test_list = self.pp.split_train_val_test(noise_free_list, 1-TEST_SPLIT,0,TEST_SPLIT)
        self.pp.dir_for_train_val_test(train_list, val_list, test_list)
        train_path, test_path = self.pp.new_csv(train_list, test_list)

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
                                              rotation_range=10,
                                              # width_shift_range=0.2,
                                              # height_shift_range=0.2,
                                              # shear_range=0.1,
                                              zoom_range=0.2,
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

    def setup_cnn_model(self):
        print "{} Model setup".format(self.feature_tested)
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), input_shape=(INPUT_DIM, INPUT_DIM, 3)))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(32, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(256))
        self.model.add(Activation('relu'))

        self.model.add(Dropout(0.5))
        self.model.add(Dense(256))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        
        # optimizer = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
        # optimizer = optimizers.Adam(lr=0.0001)
        optimizer = optimizers.RMSprop(lr=0.0001, decay=1e-6)
        # optimizer = optimizers.Adagrad(lr=0.001, epsilon=None, decay=0.0)

        if self.multiclass:
            self.model.add(Dense(self.output_dim, activation='softmax'))
            loss = 'categorical_crossentropy'
        else:
            self.model.add(Dense(self.output_dim, activation='sigmoid'))
            loss = 'binary_crossentropy'
            
        self.model.compile(optimizer, loss=loss, metrics=["accuracy"])

    def setup_mlp_model(self):
        self.model = Sequential()
        self.model.add(Flatten(input_shape=(INPUT_DIM, INPUT_DIM, 3)))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.5))

        optimizer = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
        # optimizer = optimizers.Adam(lr=0.0001)
        # optimizer = optimizers.RMSprop(lr=0.0001, decay=1e-6)
        # optimizer = optimizers.Adagrad(lr=0.0001, epsilon=None, decay=0.0)

        if self.multiclass:
            self.model.add(Dense(self.output_dim, activation='softmax'))
            loss = 'categorical_crossentropy'
        else:
            self.model.add(Dense(self.output_dim, activation='sigmoid'))
            loss = 'binary_crossentropy'

        self.model.compile(loss=loss,
                           optimizer=optimizer,
                           metrics=['accuracy'])

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

        analysis = self.model.fit_generator(generator=self.train_generator,
                                            steps_per_epoch=STEP_SIZE_TRAIN,
                                            validation_steps=STEP_SIZE_VALID,
                                            validation_data=self.valid_generator,
                                            callbacks=early_stopping,
                                            verbose=2,
                                            workers=4,
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
        count = 0
        for i in range(len(truth_list)):
            if truth_list[i] == predictions[i]:
                count += 1
        accuracy = count/float(len(truth_list))
        print "Accuracy: {:.2f}%".format(accuracy*100)
        return accuracy

    def save_csv(self, dataframe, accuracy):
        # Saving Results into csv
        csv_file = "results/{}_{}_{}_results_{}.csv".format(self.feature_tested, EPOCH_SIZE, INPUT_DIM, self.suffix)
        dataframe.to_csv(csv_file,index=False)

        data = open(csv_file, 'r').readlines()[1:]
        data.insert(0, '{},,\n'.format(accuracy))
        open(csv_file, 'w').writelines(data)


if __name__ == "__main__":

    # for feature in ALL_CLASSIFICATION:

    # for feature in ['eyeglasses', 'smiling', 'young', 'human']:
    for feature in ['human']:
        cnn = Cnn(feature, augment=False, suffix='mlp_no_aug_3layer_sgd')
        cnn.call_preprocess(shuffle=True, compress=False)
        cnn.prepare_generator()
        
        cnn.setup_mlp_model()
        # cnn.setup_cnn_model()
        cnn.model.summary()

        history = cnn.train_model()
        Plotting.plot_history(history, '', EPOCH_SIZE, cnn.feature_tested, cnn.suffix, save=True, show=False)
        cnn.evaluate_model(cnn.valid_generator)
        # # results = cnn.evaluate_model(cnn.test_generator)
        # cnn.saving_model()

        dataframe = cnn.predict_model(cnn.test_generator)
        accuracy = cnn.manual_check_model(dataframe)
        uti.save_csv(dataframe, accuracy, "{}_{}_{}_results_{}".format(cnn.feature_tested, EPOCH_SIZE, INPUT_DIM, cnn.suffix))
