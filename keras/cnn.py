from preprocess import Preprocess
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


input_dim = 64
epochs_size = 50
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
                target_size=(input_dim,input_dim))

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
                target_size=(input_dim,input_dim))

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
                target_size=(input_dim,input_dim))

print "Model setup"
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(input_dim,input_dim,3)))
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

history = model.fit_generator(generator=train_generator,
                                steps_per_epoch=STEP_SIZE_TRAIN,
                                validation_data=valid_generator,
                                validation_steps=STEP_SIZE_VALID,
                                epochs=epochs_size,
                                workers=4
                                )

print "Plotting"
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show(block=False)
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='bottom right')
plt.show(block=False)



print "Model evaluating..."
scores = model.evaluate_generator(generator=valid_generator,steps=STEP_SIZE_VALID)
print "Accuracy = ",scores[1]

model_name = "models/" + feature_tested + "_" + str(epochs_size) + "_" + str(input_dim) + "_"
# serialize model to JSON
model_json = model.to_json()
with open(model_name + "model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(model_name + "model.h5")
print("Saved model to disk")


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

csv_file = "results/" + feature_tested + "_" + str(epochs_size) + "_" + str(input_dim) + "_results.csv"
results.to_csv(csv_file,index=False)

file = open(csv_file, 'r')
data = file.readlines()[1:]
data.insert(0, '{},,\n'.format(accuracy))
file = open(csv_file, 'w')
file.writelines(data)

plt.show()
