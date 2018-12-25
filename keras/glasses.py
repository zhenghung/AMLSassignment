from preprocess import Preprocess
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
import pandas as pd
import numpy as np
import os

epochs_size = 10
feature_tested = 'hair_color'
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
datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.25)

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
                target_size=(32,32))

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
                target_size=(32,32))

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
                target_size=(32,32))

print "Model setup"
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(32,32,3)))
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
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(output_dim, activation='softmax'))
model.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6),loss="categorical_crossentropy",metrics=["accuracy"])

print "Model fitting..."
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=epochs_size
                    )

print "Model evaluating..."
model.evaluate_generator(generator=valid_generator,steps=STEP_SIZE_VALID)

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
results['Truth'] = truth_list

predictions = list(results['Predictions'])
count = 0
for i in range(len(truth_list)):
    if truth_list[i] == predictions[i]:
        count+=1
accuracy = 100*count/float(len(truth_list))
print "Accuracy: {:.2f}%".format(accuracy)

acc_list = [accuracy]
for i in range(len(truth_list)-1):
    acc_list.append(None)
results['Accuracy'] = acc_list

results.to_csv("./results/" + feature_tested + "_" + str(epochs_size) + "_results.csv",index=False)


