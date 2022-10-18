
from keras.models import load_model
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.image import imread
import pandas as pd

data_dir = "dataset"

CategoriesNames = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)',
            2:'Speed limit (50km/h)',
            3:'Speed limit (60km/h)',
            4:'Speed limit (70km/h)',
            5:'Speed limit (80km/h)',
            6:'End of speed limit (80km/h)',
            7:'Speed limit (100km/h)',
            8:'Speed limit (120km/h)',
            9:'No passing',
            10:'No passing veh over 3.5 tons',
            11:'Right-of-way at intersection',
            12:'Priority road',
            13:'Yield',
            14:'Stop',
            15:'No vehicles',
            16:'Veh > 3.5 tons prohibited',
            17:'No entry',
            18:'General caution',
            19:'Dangerous curve left',
            20:'Dangerous curve right',
            21:'Double curve',
            22:'Bumpy road',
            23:'Slippery road',
            24:'Road narrows on the right',
            25:'Road work',
            26:'Traffic signals',
            27:'Pedestrians',
            28:'Children crossing',
            29:'Bicycles crossing',
            30:'Beware of ice/snow',
            31:'Wild animals crossing',
            32:'End speed + passing limits',
            33:'Turn right ahead',
            34:'Turn left ahead',
            35:'Ahead only',
            36:'Go straight or right',
            37:'Go straight or left',
            38:'Keep right',
            39:'Keep left',
            40:'Roundabout mandatory',
            41:'End of no passing',
            42:'End no passing veh > 3.5 tons' }

model_path = "ouput_model_training/secondModel/model.h5"
loaded_model = load_model(model_path)

"""Prediction on single image """

# img = cv2.imread("dataset/Test/00001.png")
#
# image_fromarray = Image.fromarray(img, 'RGB')
# resize_image = image_fromarray.resize((30, 30))
#
# expand_input = np.expand_dims(resize_image,axis=0)
# input_data = np.array(expand_input)
# input_data = input_data/255
#
# pred = loaded_model.predict(input_data)
# result = pred.argmax()
# print(CategoriesNames[result])


"""predicting 5 random images from test data"""


test = pd.read_csv(data_dir + '/Test.csv')
imgs = test["Path"].values

plt.figure(figsize=(25,25))

for i in range(1,6):
    plt.subplot(1,5,i)
    random_img_path = data_dir + '/' + random.choice(imgs)
    rand_img = imread(random_img_path)

    image_fromarray = Image.fromarray(rand_img, 'RGB')
    resize_image=image_fromarray.resize((30, 30))

    expand_input=np.expand_dims(resize_image, axis=0)
    input_data=np.array(expand_input)
    input_data=input_data / 255

    pred=loaded_model.predict(input_data)
    result=pred.argmax()

    plt.imshow(rand_img)
    plt.xlabel(CategoriesNames[result],  color="r")
plt.show()