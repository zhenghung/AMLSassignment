import os
import random
import pandas as pd
from shutil import copy
import cv2


class Preprocess:
    def __init__(self, shuffle, compress, compress_size):
        self.shuffle = shuffle
        self.compress = compress
        self.compress_size = compress_size
        current_dir = os.path.dirname(os.path.realpath(__file__))
        self.dataset_dir = os.path.abspath(os.path.join(current_dir, "..", "AMLS_Assignment_Dataset"))
        self.images_dir = os.path.join(self.dataset_dir, 'dataset')
        self.labels_path = os.path.join(self.dataset_dir, 'attribute_list.csv')
        self.filtered_labels_path = os.path.join(self.dataset_dir, 'attribute_list_new.csv')

    def filter_noise(self):
        """
        Returns list of image names of real data (not background noise)
        """
        if os.path.isfile(self.labels_path):
            real_data = []
            labels_file = open(self.labels_path, 'r')
            lines = labels_file.readlines()
            specified_labels = {line.split(',')[0]: [int(x) for x in line.split(',')][1:] for line in lines[2:]}
            for key in specified_labels:
                if specified_labels[key] != [-1, -1, -1, -1, -1]:
                    real_data.append(key)
        else:
            print "Labels Path not valid: ",
            print self.labels_path
            return -1

        return real_data

    def new_csv(self, train_list, test_list):
        arr = [train_list, test_list]
        str_arr = ['attribute_list_train.csv', 'attribute_list_test.csv']

        if self.shuffle is not True:
            return os.path.join(self.dataset_dir, str_arr[0]), os.path.join(self.dataset_dir, str_arr[1])

        for i in range(len(arr)):
            file = pd.read_csv(self.labels_path, header=None)
            for index, row in file.iterrows():
                if index <= 1:
                    continue
                if row[0] not in arr[i]:
                    file.drop(index, axis=0, inplace=True)

            file.to_csv(os.path.join(self.dataset_dir, str_arr[i]), index=False)

            file = open(os.path.join(self.dataset_dir, str_arr[i]), 'r')
            data = file.readlines()[3:]
            file = open(os.path.join(self.dataset_dir, str_arr[i]), 'w')
            file.writelines(data)

        return os.path.join(self.dataset_dir, str_arr[0]), os.path.join(self.dataset_dir, str_arr[1])

    def split_train_val_test(self, data_list, train_ptg, val_ptg, test_ptg, randomize=True):
        """
        Splits the real_data list from filter_noise into training, validation and testing
        By default, randomize is True
        """
        size = len(data_list)
        if randomize:
            train_list = random.sample(data_list, int(train_ptg * size))
            [data_list.remove(x) for x in train_list]
            val_list = random.sample(data_list, int(val_ptg * size))
            [data_list.remove(x) for x in val_list]
            test_list = data_list
        else:
            train_list = data_list[:int(size * train_ptg)]
            [data_list.remove(x) for x in train_list]
            val_list = data_list[:int(size * val_ptg)]
            [data_list.remove(x) for x in val_list]
            test_list = data_list

        return train_list, val_list, test_list

    def dir_for_train_val_test(self, train_list, val_list, test_list):
        """
        Creates a new directory for training, validation and testing
        Then copies the selected images from the dataset into them
        """
        if os.path.isdir(os.path.join(self.dataset_dir, "training")):
            [os.remove(os.path.join(self.dataset_dir, "training", file)) for file in os.listdir(os.path.join(self.dataset_dir, "training"))]
            [os.remove(os.path.join(self.dataset_dir, "validation", file)) for file in os.listdir(os.path.join(self.dataset_dir, "validation"))]
            [os.remove(os.path.join(self.dataset_dir, "testing", file)) for file in os.listdir(os.path.join(self.dataset_dir, "testing"))]
        else:
            os.makedirs(os.path.join(self.dataset_dir, "training"))
            os.makedirs(os.path.join(self.dataset_dir, "validation"))
            os.makedirs(os.path.join(self.dataset_dir, "testing"))

        for img in os.listdir(self.images_dir):
            img_name = img.split(".")[0]

            if self.compress:
                image = cv2.imread(os.path.join(self.images_dir, img))
                resized_image = cv2.resize(image, (self.compress_size, self.compress_size))

                if img_name in train_list:
                    cv2.imwrite(os.path.join(self.dataset_dir, "training", img), resized_image)
                elif img_name in val_list:
                    cv2.imwrite(os.path.join(self.dataset_dir, "validation", img), resized_image)
                elif img_name in test_list:
                    cv2.imwrite(os.path.join(self.dataset_dir, "testing", img), resized_image)
            else:
                if img_name in train_list:
                    copy(os.path.join(self.images_dir, img), os.path.join(self.dataset_dir, "training", img))
                elif img_name in val_list:
                    copy(os.path.join(self.images_dir, img), os.path.join(self.dataset_dir, "validation", img))
                elif img_name in test_list:
                    copy(os.path.join(self.images_dir, img), os.path.join(self.dataset_dir, "testing", img))

        return 0


if __name__ == "__main__":
    inst = Preprocess(True, True)
    data_list = inst.filter_noise()
    train_list, val_list, test_list = inst.split_train_val_test(data_list, 0.8, 0, 0.2)
    c = inst.new_csv(train_list, test_list)
    # train_list, val_list, test_list = inst.split_train_val_test(data_list, 0.8, 0, 0.2)
    # inst.dir_for_train_val_test(train_list, val_list, test_list)
