# Applied Machine Learning Systems
The following documentation outlines the steps required to run the AMLSassigment (Applied Machine Learning Systems project).

## Machine Specs
Below lists the specifications of the machine used to train the models and run the predictions
```buildoutcfg
- Intel Core i5-8300H
- NVIDIA Geforce GTX1050 - 4GB
- 12GB RAM
```

# Contents

- Clone Repository
- Project structure
- Guide to Training Models
    - SVM
    - ConvNet and MLP


## Clone Repository

Access the project folder by cloning the project repository as follows:

```
$ git clone https://github.com/zhenghung/AMLSassignment.git

$ git checkout master
```

Download and extract the datasets into the repository from [here](https://drive.google.com/open?id=1NgP2jQakFHibIhpevDLshodWw-L52yXi).


## Libraries Used
1. [Tensorflow](https://www.tensorflow.com/)
2. [Keras](https://keras.io/)
3. [Scikit-learn](https://scikit-learn.org/stable/)
4. [OpenCV](https://opencv.org/)
5. [Matplotlib](https://matplotlib.org/)
6. [dlib](http://dlib.net/)
7. [numpy](http://www.numpy.org/)
8. [pandas](https://pandas.pydata.org/)

Note: Tensorflow GPU has potential for better performance during training if system has a dedicated graphics card. Please ensure prerequisites for tensorflow-gpu is met before pip installing it.
```buildoutcfg
$ sudo pip install tensorflow<-gpu> keras sklearn dlib matplotlib opencv-python numpy pandas 
```

## Project Structure
```buildoutcfg
./
├── AMLS_Assignment_Dataset
├── analysis
│   └── keras_results.xlsx
├── cnn
│   ├── cnn.py
│   ├── load_model.py
│   ├── models/
│   ├── plots/
│   └── results/
├── README.md
├── svm
│   ├── classification.py
│   ├── dlib_extractor.py
│   ├── features_and_labels/
│   │   ├── eyeglasses_labels.npy
│   │   ├── face_features.npy
│   │   ├── hair_color_labels.npy
│   │   ├── human_labels.npy
│   │   ├── smiling_labels.npy
│   │   └── young_labels.npy
│   ├── plots/
│   ├── results/
│   └── shape_predictor_68_face_landmarks.dat
├── testing_predictions
│   ├── dataset/
│   ├── results/
│   └── models/
└── tools
    ├── plotting.py
    ├── preprocess.py
    └── utils.py
```
- ```AMLS_Assignment_Dataset``` contains all datasets and labels provided
- ```analysis``` contains an xlsx format file tabulating the results of the tests
- ```cnn``` contains the classes using keras for building the models, including mlp and ConvNets
- ```svm``` contains the classes for using SVM from scikit-learn library and dlib
- ```testing_predictions``` contains files for the testing_dataset, models contain the json and h5 files, results contain the csv predictions
- ```tools``` contains classes for preprocessing the dataset, plotting the curve and saving into csv files

## Guide to building models and Training them

### SVM

In the main repository directory ./, open the terminal and run the python module

```buildoutcfg
$ python -m svm.classification
```

Results are saved ```svm/results```

### ConvNet and MLP

Results are saved in ```cnn/results```
Plots are saved in ```cnn/plots```
Models are saved in ``` cnn/models```

#### Example
To change the models, open ```cnn/cnn.py``` with a text editor and scroll down to the main section.

Uncomment one of the method calls for chosen models.

```
419             # cnn.setup_mlp_model('sgd')
420             # cnn.setup_cnn_model('adam')
421             # cnn.setup_cnn5_no_fc(opt)
422             # cnn.setup_mobilenetv2()
```
Modify the suffix parameter in line 413 to save the results and models with a different name

```buildoutcfg
413             cnn = Cnn(feature, augment=True, suffix='cnn3+fc-FINAL')
```

#### Running training
In the main repository directory ./, open the terminal and run the python module

```buildoutcfg
$ python -m cnn.cnn
```

#### Write your own main
```buildoutcfg
from cnn.cnn import cnn
from tools.preprocess import preprocess

# preprocessing
cnn = Cnn(feature, augment=True, suffix='cnn3+fc-FINAL')
cnn.call_preprocess(shuffle=False, compress=False, compress_size=INPUT_DIM)
cnn.prepare_generator()

# Building your model
cnn.setup_cnn_model('adam')
cnn.model.summary()

# Evaluating and post processing
history = cnn.train_model()
Plotting.plot_history(history, '', EPOCH_SIZE, cnn.feature_tested, cnn.suffix, save=True, show=False)
cnn.evaluate_model(cnn.valid_generator)
cnn.saving_model()

dataframe = cnn.predict_model(cnn.test_generator)
accuracy, f1 = cnn.manual_check_model(dataframe)
print 'f1 score:', f1
cnn.save_csv(dataframe, accuracy)

```