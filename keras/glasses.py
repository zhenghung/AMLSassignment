import pandas as pd
from preprocess import Preprocess
from keras.preprocessing.image import ImageDataGenerator

pp = Preprocess()

dataframe = pd.read_csv(pp.labels_path, names=['hair_color','eyeglasses','smiling','young','human'])
print len(list(dataframe['eyeglasses'])[2:])
