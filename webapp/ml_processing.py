import os
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image

def prediction():
    model_new = tf.keras.models.load_model("savedmodel/skin_cancer_classifier.hdf5")
    imagesname = os.listdir("media")
    imagespath = []
    for  i in range(len(imagesname)):
        imagespath.append("media/" + str(imagesname[i])  )
        #print(imagespath)
    for i in range(len(imagespath)):
        test_image = image.load_img( imagespath[i] , target_size = (100, 100 , 3))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image , axis = 0)
        result = model_new.predict(test_image)
        print(result)

    pass

    

#this function ensure to delete all the data created by user 
def delete():
    #code to delete files from media directory
    media_dir = os.path.join(BASE_DIR, 'media')
    for f in os.listdir(media_dir):
        os.remove(os.path.join(media_dir, f))