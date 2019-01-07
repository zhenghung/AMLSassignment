
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from keras import optimizers
# from preprocess import Preprocess
import numpy as np
import os
import pandas as pd
from tools.utils import Utilities as uti

current_dir = os.path.dirname(os.path.realpath(__file__))

ALL_CLASSIFICATION = ['hair_color', 'eyeglasses', 'smiling', 'young', 'human']

test_datagen = ImageDataGenerator(rescale=1./255.)
test_generator = test_datagen.flow_from_directory(directory=os.path.join(current_dir, "testing_dataset"),
                                                  target_size=(256, 256),
                                                  color_mode="rgb",
                                                  class_mode=None,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  seed=42,
                                                  )

for feature_tested in ['hair_color']:
    # load json and create model
    json_file = open('models/{}_cnn3fc_model.json'.format(feature_tested), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("models/{}_cnn3fc_model.h5".format(feature_tested))
    print("Loaded model from disk")
    # loaded_model.summary()
    optimizer = optimizers.Adam(lr=0.0001)
    loss = 'categorical_crossentropy'

    loaded_model.compile(optimizer, loss=loss, metrics=["accuracy"])
    test_generator.reset()

    pred = loaded_model.predict_generator(test_generator, verbose=1, steps=test_generator.n//test_generator.batch_size)

    predicted_class_indices = np.argmax(pred, axis=1)
    labels = [0,1,2,3,4,5]
    labels = dict((v, k) for k, v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]

    results = uti.to_dataframe(test_generator.filenames, predictions)

'''

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
'''