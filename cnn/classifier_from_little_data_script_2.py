
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from preprocess import Preprocess
import pandas as pd
import os
# dimensions of our images.
INPUT_DIM = 128
img_width, img_height = INPUT_DIM, INPUT_DIM

top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = '/../AMLS_Assignment_Dataset/training'
validation_data_dir = '/../AMLS_Assignment_Dataset/testing'
nb_train_samples = 3652
nb_validation_samples = 913
epochs = 50
batch_size = 16
ALL_CLASSIFICATION = ['hair_color', 'eyeglasses', 'smiling','young','human']

TEST_SPLIT=0.2
feature_tested = 'smiling'

def save_bottlebeck_features():

    pp = Preprocess(False)
    noise_free_list = pp.filter_noise()
    train_list, val_list, test_list = pp.split_train_val_test(noise_free_list, 1-TEST_SPLIT,0,TEST_SPLIT)
    pp.dir_for_train_val_test(train_list, val_list, test_list)
    train_path, test_path = pp.new_csv(train_list, test_list)

    traindf = pd.read_csv(train_path, names=['file_name']+ALL_CLASSIFICATION)
    testdf = pd.read_csv(test_path, names=['file_name']+ALL_CLASSIFICATION)


    # datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    datagen=ImageDataGenerator(rescale=1./255.,
                                    validation_split=0.25,
                                    rotation_range=20,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.1,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    fill_mode='nearest')

    train_generator=datagen.flow_from_dataframe(
                    dataframe=traindf,
                    directory=os.path.join(pp.dataset_dir, 'training'),
                    x_col="file_name",
                    y_col=feature_tested,
                    has_ext=False,                                     
                    subset="training",
                    batch_size=batch_size,
                    seed=42,
                    shuffle=True,
                    class_mode="categorical",
                    target_size=(INPUT_DIM,INPUT_DIM))

    # generator = datagen.flow_from_directory(
    #     train_data_dir,
    #     target_size=(img_width, img_height, 3),
    #     batch_size=batch_size,
    #     class_mode=None,
    #     shuffle=False)
    # bottleneck_features_train = model.predict_generator(
    #     train_generator, train_generator.n // batch_size)
    # np.save(open('bottleneck_features_train.npy', 'w'),
    #         bottleneck_features_train)

    # generator = datagen.flow_from_directory(
    #     validation_data_dir,
    #     target_size=(img_width, img_height, 3),
    #     batch_size=batch_size,
    #     class_mode=None,
    #     shuffle=False)

    valid_generator=datagen.flow_from_dataframe(
                dataframe=traindf,
                directory=os.path.join(pp.dataset_dir, 'training'),
                x_col="file_name",
                y_col=feature_tested,
                has_ext=False,
                subset="validation",
                batch_size=batch_size,
                seed=42,
                shuffle=True,
                class_mode="categorical",
                target_size=(INPUT_DIM,INPUT_DIM))

    bottleneck_features_validation = model.predict_generator(
        valid_generator, valid_generator.n // batch_size)
    np.save(open('bottleneck_features_validation.npy', 'w'),
            bottleneck_features_validation)


def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy'))
    train_labels = np.array(
        [0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2))

    validation_data = np.load(open('bottleneck_features_validation.npy'))
    validation_labels = np.array(
        [0] * (nb_validation_samples / 2) + [1] * (nb_validation_samples / 2))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)


save_bottlebeck_features()
train_top_model()
