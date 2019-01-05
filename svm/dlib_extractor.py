import os
import numpy as np
from keras.preprocessing import image
import cv2
import dlib
import time
import cv2
import matplotlib.pyplot as plt
global basedir, image_paths, target_size
basedir = './../AMLS_Assignment_Dataset'
images_dir = os.path.join(basedir, 'dataset')
labels_filename = 'attribute_list.csv'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


# how to find frontal human faces in an image using 68 landmarks.  These are points on the face such as the corners of the mouth, along the eyebrows, on the eyes, and so forth.

# The face detector we use is made using the classic Histogram of Oriented
# Gradients (HOG) feature combined with a linear classifier, an image pyramid,
# and sliding window detection scheme.  The pose estimator was created by
# using dlib's implementation of the paper:
# One Millisecond Face Alignment with an Ensemble of Regression Trees by
# Vahid Kazemi and Josephine Sullivan, CVPR 2014
# and was trained on the iBUG 300-W face landmark dataset (see https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/):
#     C. Sagonas, E. Antonakos, G, Tzimiropoulos, S. Zafeiriou, M. Pantic.
#     300 faces In-the-wild challenge: Database and results.
#     Image and Vision Computing (IMAVIS), Special Issue on Facial Landmark Localisation "In-The-Wild". 2016.


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def run_dlib_shape(image):
    # in this function we load the image, detect the landmarks of the face, and then return the image and the landmarks
    # load the input image, resize it, and convert it to grayscale
    resized_image = image.astype('uint8')

    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype('uint8')

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    num_faces = len(rects)

    if num_faces == 0:
        return None, resized_image

    face_areas = np.zeros((1, num_faces))
    face_shapes = np.zeros((136, num_faces), dtype=np.int64)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        temp_shape = predictor(gray, rect)
        temp_shape = shape_to_np(temp_shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)],
        #   (x, y, w, h) = face_utils.rect_to_bb(rect)
        (x, y, w, h) = rect_to_bb(rect)
        face_shapes[:, i] = np.reshape(temp_shape, [136])
        face_areas[0, i] = w * h
    # find largest face and keep
    dlibout = np.reshape(np.transpose(face_shapes[:, np.argmax(face_areas)]), [68, 2])

    return dlibout, resized_image

def extract_edges(img_path):
    image = cv2.imread(img_path)
    img = cv2.resize(image, (32, 32))
    edges = cv2.Canny(img, img.shape[0], img.shape[1])

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

    plt.show()

    coordinates = []
    for row in range(len(edges)):
        for column in range(len(edges[0])):
            if edges[row][column] == 255:
                coordinates.append([row, column])

    print len(coordinates)
    return coordinates


def extract_features_labels(data_list):
    """
    This funtion extracts the landmarks features for all images in the folder 'dataset/celeba'.
    It also extract the gender label for each image.
    :return:
        landmark_features:  an array containing 68 landmark points for each image in which a face was detected
        gender_labels:      an array containing the gender label (male=0 and female=1) for each image in
                            which a face was detected
    """

    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    target_size = (32,32,3)
    labels_file = open(os.path.join(basedir, labels_filename), 'r')
    lines = labels_file.readlines()
    hair_color_labels = {line.split(',')[0] : int(line.split(',')[1]) for line in lines[2:]}
    eyeglasses_labels = {line.split(',')[0] : int(line.split(',')[2]) for line in lines[2:]}
    smiling_labels = {line.split(',')[0] : int(line.split(',')[3]) for line in lines[2:]}
    young_labels = {line.split(',')[0] : int(line.split(',')[4]) for line in lines[2:]}
    human_labels = {line.split(',')[0] : int(line.split(',')[5]) for line in lines[2:]}

    all_features = []
    all_images = []
    all_hair_color_labels = []
    all_eyeglasses_labels = []
    all_smiling_labels = []
    all_young_labels = []
    all_human_labels = []

    if os.path.isdir(images_dir):
        count=0
        percent = 0
        timeA = time.time()
        for file_name in data_list:
            # file_name= img_path.split('.')[-2].split('/')[-1]
            img_path = os.path.join(basedir, 'dataset',file_name + '.png')
            
            # Progress Bar
            timeB = timeA
            timeA = time.time()
            dur_per_count = timeA-timeB
            eta_str = "ETA: {}s".format(int(dur_per_count*(len(data_list)-count)))
            count+=1
            cur_ptg = int(100*count/float(len(data_list)))
            if len(str(cur_ptg))<3:
                str_ptg = ' '*(3-len(str(cur_ptg)))+str(cur_ptg)
            else:
                str_ptg = cur_ptg

            progress = int(cur_ptg*0.5)
            if cur_ptg > percent:
                print 'Percentage Done: {}%  [{}{}{}]  {}'.format(str_ptg,'#'*progress, '>','-'*(50-progress),eta_str)
                percent = cur_ptg

            # load image
            img = image.img_to_array(
                image.load_img(img_path,
                               target_size=None,
                               interpolation='bicubic'))
            features, _ = run_dlib_shape(img)

            # edges = extract_edges(img_path)


            # img = image.img_to_array(
            #     image.load_img(img_path,
            #                    target_size=target_size,
            #                    interpolation='bicubic'))
            if features is not None:
                all_features.append(features)
                # all_images.append(img)
                all_hair_color_labels.append(hair_color_labels[file_name])
                all_eyeglasses_labels.append(eyeglasses_labels[file_name])
                all_smiling_labels.append(smiling_labels[file_name])
                all_young_labels.append(young_labels[file_name])
                all_human_labels.append(human_labels[file_name])

    landmark_features = np.array(all_features)
    # ds_images = np.array(all_images)
    hair_color_labels = np.array(all_hair_color_labels) # simply converts the -1 into 0, so male=0 and female=1
    eyeglasses_labels = np.array(all_eyeglasses_labels)
    smiling_labels = np.array(all_smiling_labels)
    young_labels = np.array(all_young_labels)
    human_labels = np.array(all_human_labels)

    NPY_FILE_DIR = 'features_and_labels/'
    np.save(NPY_FILE_DIR+'face_features', landmark_features)
    # np.save(NPY_FILE_DIR+'ds_images', ds_images)
    np.save(NPY_FILE_DIR+'hair_color_labels', hair_color_labels)
    np.save(NPY_FILE_DIR+'eyeglasses_labels', eyeglasses_labels)
    np.save(NPY_FILE_DIR+'smiling_labels', smiling_labels)
    np.save(NPY_FILE_DIR+'young_labels', young_labels)
    np.save(NPY_FILE_DIR+'human_labels', human_labels)

    return 0

def load_features_extract_labels(features, labels):
    """
    Loads the existing npy files for features and labels
    """
    landmark_features = np.load(features)
    specific_labels = np.load(labels)
    
    return landmark_features, specific_labels