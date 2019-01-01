from preprocess import Preprocess
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
from plotting import Plotting
# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

ALL_CLASSIFICATION = ['hair_color', 'eyeglasses', 'smiling','young','human']

EPOCH_SIZE = 10
INPUT_DIM = 128
BATCH_SIZE = 32
TEST_SPLIT = 0.2

class Cnn():

    def __init__(self, feature_tested, neuron_size):
        self.feature_tested = feature_tested
        self.neuron_size = neuron_size
        self.multiclass = True
        if feature_tested == 'hair_color':
            # self.multiclass = True
            self.output_dim = 7
        else:
            # self.multiclass = False
            self.output_dim = 2

    def call_preprocess(self):
        print "Preprocessing Dataset.."
        self.pp = Preprocess()
        self.noise_free_list = self.pp.filter_noise()
        train_list, val_list, test_list = self.pp.split_train_val_test(self.noise_free_list, 1-TEST_SPLIT,0,TEST_SPLIT)
        self.pp.dir_for_train_val_test(train_list, val_list, test_list)
        train_path, test_path = self.pp.new_csv(train_list, test_list)

        self.traindf = pd.read_csv(train_path, names=['file_name']+ALL_CLASSIFICATION)
        self.testdf = pd.read_csv(test_path, names=['file_name']+ALL_CLASSIFICATION)
        print self.traindf.head()

    def prepare_generator(self):
        self.datagen=ImageDataGenerator(rescale=1./255.,
                                        validation_split=0.25,
                                        rotation_range=20,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        shear_range=0.1,
                                        zoom_range=0.2,
                                        horizontal_flip=True,
                                        fill_mode='nearest')

        self.train_generator=self.datagen.flow_from_dataframe(
                        dataframe=self.traindf,
                        directory=os.path.join(self.pp.dataset_dir, 'training'),
                        x_col="file_name",
                        y_col=self.feature_tested,
                        has_ext=False,                                     
                        subset="training",
                        batch_size=BATCH_SIZE,
                        seed=42,
                        shuffle=True,
                        class_mode="categorical",
                        target_size=(INPUT_DIM,INPUT_DIM))

        self.valid_generator=self.datagen.flow_from_dataframe(
                        dataframe=self.traindf,
                        directory=os.path.join(self.pp.dataset_dir, 'training'),
                        x_col="file_name",
                        y_col=self.feature_tested,
                        has_ext=False,
                        subset="validation",
                        batch_size=BATCH_SIZE,
                        seed=42,
                        shuffle=True,
                        class_mode="categorical",
                        target_size=(INPUT_DIM,INPUT_DIM))

        self.test_datagen=ImageDataGenerator(rescale=1./255.)

        self.test_generator=self.test_datagen.flow_from_dataframe(
                        dataframe=self.testdf,
                        directory=os.path.join(self.pp.dataset_dir, 'testing'),
                        x_col="file_name",
                        y_col=self.feature_tested,
                        has_ext=False,
                        batch_size=1,
                        seed=42,
                        shuffle=False,
                        class_mode="categorical",
                        target_size=(INPUT_DIM,INPUT_DIM))

    def setup_cnn_model(self):
        print "Model setup"
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), padding='same',
                         input_shape=(INPUT_DIM,INPUT_DIM,3)))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(32, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(64, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(256))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.output_dim, activation='softmax'))
        self.model.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6),
                        loss="categorical_crossentropy",
                        metrics=["accuracy"])

    def setup_cnn_2_model(self):
        print "Model setup"
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), input_shape=(INPUT_DIM,INPUT_DIM,3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(32, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(self.neuron_size))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.output_dim))
        self.model.add(Activation('softmax'))
        self.model.compile(loss='categorical_crossentropy',
                              optimizer='rmsprop',
                              metrics=['accuracy'])

    def setup_mlp_model(self):
        self.model = Sequential()
        # Dense(64) is a fully-connected layer with 64 hidden units.
        # in the first layer, you must specify the expected input data shape:
        # here, 20-dimensional vectors.
        self.model.add(Flatten(input_shape=(INPUT_DIM,INPUT_DIM,3)))
        self.model.add(Dense(128, activation='relu'))
        # self.model.add(Dropout(0.5))
        self.model.add(Dense(128, activation='relu'))
        # self.model.add(Dropout(0.5))

        if self.multiclass:
            self.model.add(Dense(self.output_dim, activation='softmax'))
            sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
            loss = 'categorical_crossentropy'
            optimizer = sgd
        else:
            self.model.add(Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
            optimizer = 'rmsprop'

        self.model.compile(loss=loss,
                          optimizer=optimizer,
                          metrics=['accuracy'])

    def train_model(self):
        print "Model fitting..."
        STEP_SIZE_TRAIN=self.train_generator.n//self.train_generator.batch_size
        STEP_SIZE_VALID=self.valid_generator.n//self.valid_generator.batch_size

        history = self.model.fit_generator(generator=self.train_generator,
                                            steps_per_epoch=STEP_SIZE_TRAIN,
                                            validation_steps=STEP_SIZE_VALID,
                                            validation_data=self.valid_generator,
                                            epochs=EPOCH_SIZE,
                                            workers=4
                                            )
        return history

    def evaluate_model(self, generator):
        print "Model evaluating..."
        generator.reset()
        step_size = generator.n//generator.batch_size
        scores = self.model.evaluate_generator(generator=generator, verbose = 1, steps=step_size)
        print "Loss: ", scores[0]
        print "Accuracy: ",scores[1]
        return scores


    def saving_model(self):
        # Saving Model
        model_name = "models/{}_{}_{}_".format(self.feature_tested, EPOCH_SIZE, INPUT_DIM)
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(model_name + "model_{}.json".format(self.neuron_size), "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(model_name + "model_{}.h5".format(self.neuron_size))
        print "Saved model to disk"


    def predict_model(self, test_generator):
        '''
        Predicts the output of the testing images and returns the dataframe with the output
        '''
        test_generator.reset()
        step_size = test_generator.n//test_generator.batch_size
        pred=self.model.predict_generator(test_generator, verbose=2, steps=step_size)
        predicted_class_indices=np.argmax(pred,axis=1)

        labels = (self.train_generator.class_indices)
        labels = dict((v,k) for k,v in labels.items())
        predictions = [labels[k] for k in predicted_class_indices]

        filenames=test_generator.filenames

        filenames = [int(x.split(".")[0]) for x in filenames]
        results=pd.DataFrame({"Filename":filenames,
                              "Predictions":predictions})
        results = results.sort_values('Filename')

        return results


    def manual_check_model(self, results):
        truth_list = list(self.testdf[self.feature_tested])

        predictions = list(results['Predictions'])
        count = 0
        for i in range(len(truth_list)):
            if truth_list[i] == predictions[i]:
                count+=1
        accuracy = count/float(len(truth_list))
        print "Accuracy: {:.2f}%".format(accuracy*100)
        return accuracy



    def save_csv(self, dataframe, accuracy):
        # Saving Results into csv
        # csv_file = "results/" + feature_tested + "_" + str(EPOCH_SIZE) + "_" + str(INPUT_DIM) + "_results.csv"
        csv_file = "results/{}_{}_{}_results_{}.csv".format(self.feature_tested, EPOCH_SIZE, INPUT_DIM, self.neuron_size)
        dataframe.to_csv(csv_file,index=False)

        file = open(csv_file, 'r')
        data = file.readlines()[1:]
        data.insert(0, '{},,\n'.format(accuracy))
        file = open(csv_file, 'w')
        file.writelines(data)


if __name__=="__main__":
    # for neuron_size in [32,64,128,512]:
    if True:
        neuron_size = 128
        # ALL_CLASSIFICATION =  ['smiling','young','human']
        for feature in ['eyeglasses']:
            cnn = Cnn(feature,neuron_size)
            cnn.call_preprocess()
            cnn.prepare_generator()
            
            cnn.setup_mlp_model()
            # cnn.setup_cnn_model()
            # cnn.setup_cnn_2_model()

            history = cnn.train_model()
            Plotting.plot_history(history, '', EPOCH_SIZE, cnn.feature_tested, cnn.neuron_size, False)
            cnn.evaluate_model(cnn.valid_generator)
            # results = cnn.evaluate_model(cnn.test_generator)
            # cnn.saving_model()

            dataframe = cnn.predict_model(cnn.test_generator)
            accuracy = cnn.manual_check_model(dataframe)
            # cnn.save_csv(dataframe, accuracy)









'''
# for feature_tested in ALL_CLASSIFICATION:

if True:

    INPUT_DIM = 32
    epochs_size = 1
    feature_tested = 'human'
    if feature_tested == 'hair_color':
        output_dim = 7
    else:
        output_dim = 2

    print "Preprocessing Dataset.."
    pp = Preprocess()
    noise_free_list = pp.filter_noise()
    train_list, val_list, test_list = pp.split_train_val_test(noise_free_list, 0.8,0,0.2)
    pp.dir_for_train_val_test(train_list, val_list, test_list)
    train_path, test_path = pp.new_csv(train_list, test_list)

    traindf = pd.read_csv(train_path, names=['file_name','hair_color','eyeglasses','smiling','young','human'])
    testdf = pd.read_csv(test_path, names=['file_name','hair_color','eyeglasses','smiling','young','human'])
    print traindf.head()

    print "Preparing Generators.."
    datagen=ImageDataGenerator(rescale=1./255.,
                                validation_split=0.25,
                                rotation_range=20,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                shear_range=0.1,
                                zoom_range=0.2,
                                horizontal_flip=True,
                                fill_mode='nearest')

    # datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.25)

    train_generator=datagen.flow_from_dataframe(
                    dataframe=traindf,
                    directory=os.path.join(pp.dataset_dir, 'training'),
                    x_col="file_name",
                    y_col=feature_tested,
                    has_ext=False,                                     
                    subset="training",
                    batch_size=32,
                    seed=42,
                    shuffle=True,
                    class_mode="categorical",
                    target_size=(INPUT_DIM,INPUT_DIM))

    valid_generator=datagen.flow_from_dataframe(
                    dataframe=traindf,
                    directory=os.path.join(pp.dataset_dir, 'training'),
                    x_col="file_name",
                    y_col=feature_tested,
                    has_ext=False,
                    subset="validation",
                    batch_size=32,
                    seed=42,
                    shuffle=True,
                    class_mode="categorical",
                    target_size=(INPUT_DIM,INPUT_DIM))

    test_datagen=ImageDataGenerator(rescale=1./255.)

    test_generator=test_datagen.flow_from_dataframe(
                    dataframe=testdf,
                    directory=os.path.join(pp.dataset_dir, 'testing'),
                    x_col="file_name",
                    y_col=None,
                    has_ext=False,
                    batch_size=1,
                    seed=42,
                    shuffle=False,
                    class_mode=None,
                    target_size=(INPUT_DIM,INPUT_DIM))

    print "Model setup"
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(INPUT_DIM,INPUT_DIM,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6),
                    loss="categorical_crossentropy",
                    metrics=["accuracy"])

    print "Model fitting..."
    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
    STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

    history = model.fit_generator(generator=train_generator,
                                    steps_per_epoch=STEP_SIZE_TRAIN,
                                    validation_data=valid_generator,
                                    validation_steps=STEP_SIZE_VALID,
                                    epochs=EPOCH_SIZE
                                    ,workers=8
                                    )


    # Plotting Loss and Accuracy
    print "Plotting"
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    fig1, ax1 = plt.subplots()

    ax1.plot(history.history['acc'])
    ax1.plot(history.history['val_acc'])
    ax1.set_title('model accuracy')
    ax1.set_ylabel('accuracy')
    ax1.set_xlabel('epoch')
    ax1.legend(['train', 'validation'], loc='lower right')
    plt.show(block=False)
    plt.savefig('plots/{}_{}_accuracy.png'.format(EPOCH_SIZE,feature_tested))

    # summarize history for loss
    fig2, ax2 = plt.subplots()

    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('model loss')
    ax2.set_ylabel('loss')
    ax2.set_xlabel('epoch')
    ax2.legend(['train', 'validation'], loc='upper right')
    plt.show(block=False)
    plt.savefig('plots/{}_{}_loss.png'.format(EPOCH_SIZE,feature_tested))


    print "Model evaluating..."
    scores = model.evaluate_generator(generator=valid_generator, verbose = 1, steps=STEP_SIZE_VALID)
    print "Loss: ", scores[0]
    print "Accuracy: ",scores[1]


    # Saving Model
    model_name = "models/" + feature_tested + "_" + str(EPOCH_SIZE) + "_" + str(INPUT_DIM) + "_"
    # serialize model to JSON
    model_json = model.to_json()
    with open(model_name + "model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_name + "model.h5")
    print("Saved model to disk")


    # Testing Model
    print "Model testing..."
    test_generator.reset()
    pred=model.predict_generator(test_generator,verbose=1, steps=STEP_SIZE_TEST)
    predicted_class_indices=np.argmax(pred,axis=1)

    labels = (train_generator.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]

    filenames=test_generator.filenames

    filenames = [int(x.split(".")[0]) for x in filenames]
    results=pd.DataFrame({"Filename":filenames,
                          "Predictions":predictions})
    results = results.sort_values('Filename')

    truth_list = list(testdf[feature_tested])

    predictions = list(results['Predictions'])
    count = 0
    for i in range(len(truth_list)):
        if truth_list[i] == predictions[i]:
            count+=1
    accuracy = count/float(len(truth_list))
    print "Accuracy: {:.2f}%".format(accuracy*100)


    test_datagen=ImageDataGenerator(rescale=1./255.)

    test_generator=test_datagen.flow_from_dataframe(
                    dataframe=testdf,
                    directory=os.path.join(pp.dataset_dir, 'testing'),
                    x_col="file_name",
                    y_col=feature_tested,
                    has_ext=False,
                    batch_size=1,
                    seed=42,
                    shuffle=False,
                    class_mode="categorical",
                    target_size=(INPUT_DIM,INPUT_DIM))


    print "Model evaluating..."
    scores = model.evaluate_generator(generator=test_generator, verbose = 1, steps=STEP_SIZE_TEST)
    print "Loss: ", scores[0]
    print "Accuracy: ",scores[1]


    # Saving Results into csv
    csv_file = "results/" + feature_tested + "_" + str(EPOCH_SIZE) + "_" + str(INPUT_DIM) + "_results.csv"
    results.to_csv(csv_file,index=False)

    file = open(csv_file, 'r')
    data = file.readlines()[1:]
    data.insert(0, '{},,\n'.format(accuracy))
    file = open(csv_file, 'w')
    file.writelines(data)
'''