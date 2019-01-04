from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from preprocess import Preprocess
import numpy as np
import pandas as pd
from plotting import Plotting

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#Get back the convolutional part of a VGG network trained on ImageNet
model_vgg16_conv = VGG16(weights='imagenet', include_top=False, input_shape=(64,64,3))
model_vgg16_conv.summary()

#Create your own input format (here 3x200x200)
input = Input(shape=(32,32,3),name = 'image_input')

#Use the generated model 
output_vgg16_conv = model_vgg16_conv(input)

#Add the fully-connected layers 
x = Flatten(name='flatten')(output_vgg16_conv)
x = Dense(1028, activation='relu', name='fc1')(x)
x = Dense(1028, activation='relu', name='fc2')(x)
x = Dense(2, activation='softmax', name='predictions')(x)

#Create your own model 
my_model = Model(input=input, output=x)

#In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
my_model.summary()

my_model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

##########################3

ALL_CLASSIFICATION = ['hair_color', 'eyeglasses', 'smiling','young','human']

EPOCH_SIZE = 10
INPUT_DIM = 32
BATCH_SIZE = 32
TEST_SPLIT = 0.2

pp = Preprocess()
feature_tested = 'human'
print "Preprocessing Dataset.."
pp = Preprocess()
noise_free_list = pp.filter_noise()
train_list, val_list, test_list = pp.split_train_val_test(noise_free_list, 1-TEST_SPLIT,0,TEST_SPLIT)
pp.dir_for_train_val_test(train_list, val_list, test_list)
train_path, test_path = pp.new_csv(train_list, test_list)

traindf = pd.read_csv(train_path, names=['file_name']+ALL_CLASSIFICATION)
testdf = pd.read_csv(test_path, names=['file_name']+ALL_CLASSIFICATION)
print traindf.head()



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
                batch_size=BATCH_SIZE,
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
                batch_size=BATCH_SIZE,
                seed=42,
                shuffle=True,
                class_mode="categorical",
                target_size=(INPUT_DIM,INPUT_DIM))

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



STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

history = my_model.fit_generator(generator=train_generator,
                                    steps_per_epoch=STEP_SIZE_TRAIN,
                                    validation_steps=STEP_SIZE_VALID,
                                    validation_data=valid_generator,
                                    epochs=EPOCH_SIZE,
                                    workers=4
                                    )


print "Model evaluating..."
valid_generator.reset()
step_size = valid_generator.n//valid_generator.batch_size
scores = my_model.evaluate_generator(generator=valid_generator, verbose = 1, steps=step_size)
print "Loss: ", scores[0]
print "Accuracy: ",scores[1]

plotting.plot_history(history, '', EPOCH_SIZE, feature_tested, 128, False)