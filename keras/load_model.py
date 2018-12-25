# from __future__ import print_function

from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from preprocess import Preprocess
import numpy as np
import os
import pandas as pd

cur_dir = os.path.dirname(os.path.realpath(__file__))
name = cur_dir + '/models/eyeglasses_1_64_'
feature_tested = 'eyeglasses'
# load json and create model
json_file = open(name + 'model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(name + "model.h5")
print("Loaded model from disk")
 
pp = Preprocess()
train_path = '/home/bryan/AppliedML_Assessment/AMLSassignment/AMLS_Assignment_Dataset/attribute_list_train.csv'

traindf = pd.read_csv(train_path, names=['file_name','hair_color','eyeglasses','smiling','young','human'])
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
                target_size=(64,64))
test_datagen=ImageDataGenerator(rescale=1./255.)
test_path = '/home/bryan/AppliedML_Assessment/AMLSassignment/AMLS_Assignment_Dataset/attribute_list_test.csv'
testdf = pd.read_csv(test_path, names=['file_name','hair_color','eyeglasses','smiling','young','human'])
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
                target_size=(64,64))

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.predict_generator(generator=test_generator,verbose=1,steps=STEP_SIZE_TEST)

# print score
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
predicted_class_indices=np.argmax(score,axis=1)

labels = (train_generator.class_indices)
print labels
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