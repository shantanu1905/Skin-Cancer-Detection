from django.shortcuts import render
import os
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
from django.contrib import messages
from django.contrib.messages.views import SuccessMessageMixin
# Create your views here.
from .ml_processing import delete
from PIL import Image
import numpy as np
#from webapp.skin_cancer_detection import skin_cancer_detection as SCD
#import skin_cancer_detection.py as SCD




import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D
from tensorflow.keras.models import Sequential

classes = {
    0: ("actinic keratoses and intraepithelial carcinomae(Cancer)"),
    1: ("basal cell carcinoma(Cancer)"),
    2: ("benign keratosis-like lesions(Non-Cancerous)"),
    3: ("dermatofibroma(Non-Cancerous)"),
    4: ("melanocytic nevi(Non-Cancerous)"),
    5: ("pyogenic granulomas and hemorrhage(Can lead to cancer)"),
    6: ("melanoma(Cancer)"),
}


model = Sequential()
model.add(
    Conv2D(
        16,
        kernel_size=(3, 3),
        input_shape=(28, 28, 3),
        activation="relu",
        padding="same",
    )
)
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(tf.keras.layers.BatchNormalization())
model.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))
model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(tf.keras.layers.BatchNormalization())
model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
model.add(Conv2D(256, kernel_size=(3, 3), activation="relu"))
model.add(Flatten())
model.add(tf.keras.layers.Dropout(0.2))
model.add(Dense(256, activation="relu"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.2))
model.add(Dense(128, activation="relu"))
model.add(tf.keras.layers.BatchNormalization())
model.add(Dense(64, activation="relu"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.2))
model.add(Dense(32, activation="relu"))
model.add(tf.keras.layers.BatchNormalization())
model.add(Dense(7, activation="softmax"))
model.summary()
model.load_weights("savedmodel/bestmodel.hdf5")


def Home(request):
        if request.method == 'POST' :
                myfile = request.FILES.getlist('file')
                
                for files in myfile:
                        fs = FileSystemStorage()
                        filename = fs.save(files.name, files)
                        messages.success(request, 'Files uploaded successfully')
                        
                        #size=fs.size( files)
        
               
                return render(request, 'home.html' , {})
        
        else:
                return render(request, 'home.html')

def Filedetails(request):
        imagesname = os.listdir("media")
        imagespath = []
        for  i in range(len(imagesname)):
                imagespath.append("media/" + str(imagesname[i])  )
                print(imagespath)

        context = {"datalist":imagespath}

        return render(request, 'filedetails.html' , context)



def Deletedata(request):
        if request.method == "POST":
                delete()
                print("executed")
                return render(request, 'home.html')

def Preddddiction(request):
        if request.method == "POST":
                model_new = tf.keras.models.load_model("savedmodel/bestmodel.hdf5")
                imagesname = os.listdir("media")
                imagespath = []
                result_output = []
                for  i in range(len(imagesname)):
                        imagespath.append("media/" + str(imagesname[i])  )
                        #print(imagespath)
                for i in range(len(imagespath)):
                        test_image = image.load_img( imagespath[i] , target_size = (28, 28, 3))  #for skin_cancer_classifer.hdf5 use 100x100 size
                        test_image = image.img_to_array(test_image)
                        test_image = np.expand_dims(test_image , axis = 0)
                        result = model_new.predict(test_image)
                        print(result)
                        if result[0][0]:
                                result_output.append(str(imagesname[i]) + "   " +   "Actinic keratoses (akiec)")
                                #print( str(imagesname[i]) + "     Actinic keratoses (akiec)")
                        elif result[0][1]:
                                result_output.append(str(imagesname[i]) + "   " +  " Basal cell carcinoma (bcc)")
                                #print( str(imagesname[i]) +  " Basal cell carcinoma (bcc)")
                        elif result[0][2]:
                                result_output.append(str(imagesname[i]) + "   " +  " Benign keratosis-like lesions (bkl)")
                                #print(  str(imagesname[i]) +  " Benign keratosis-like lesions (bkl)")        
                        elif result[0][3]:
                                result_output.append( str(imagesname[i]) +  "   " + " Dermatofibroma (df)")
                                #print(  str(imagesname[i]) +  " Dermatofibroma (df)")
                        elif result[0][4]:
                                result_output.append(str(imagesname[i]) +  "   " + " Melanoma (mel)")
                                #print( str(imagesname[i]) + " Melanoma (mel)")
                        elif result[0][5]:
                                result_output.append(str(imagesname[i]) + "   " + " Melanocytic nevi (nv)")
                                #print( str(imagesname[i]) + " Melanocytic nevi (nv)")
                        elif result[0][6]:
                                result_output.append(str(imagesname[i]) + "   " + " Vascular lesions (vas)")
                                #print(  str(imagesname[i]) + " Vascular lesions (vas)")
                        context = {"prediction" : result_output }




                
                
                return render(request, 'prediction.html' , context)
        


def Prediction(request):
        if request.method == "POST":
                model_new = tf.keras.models.load_model("savedmodel/bestmodel.hdf5")
                imagesname = os.listdir("media")
                imagespath = []
                result_output = []
                for  i in range(len(imagesname)):
                        imagespath.append("media/" + str(imagesname[i])  )
                        #print(imagespath[i])
                for j in range(len(imagespath)):
                        predict = []
                        print(len(imagespath))
                        print(imagespath[j])
                        inputimg = Image.open(imagespath[i])
                  
                        inputimg = inputimg.resize((28, 28))
                        img = np.array(inputimg).reshape(-1, 28, 28, 3)
                        result = model.predict(img)
                    
                        result = result.tolist()
                        #print(result)
                        max_prob = max(result[0])
                        class_ind = result[0].index(max_prob)
                       # print(class_ind)
                        result = classes[class_ind]
                       # print(result)
                        if class_ind == 0:
                                
                                info = "Actinic keratosis also known as solar keratosis or senile keratosis are names given to intraepithelial keratinocyte dysplasia. As such they are a pre-malignant lesion or in situ squamous cell carcinomas and thus a malignant lesion."
                        elif class_ind == 1:
                                info = "Basal cell carcinoma is a type of skin cancer. Basal cell carcinoma begins in the basal cells — a type of cell within the skin that produces new skin cells as old ones die off.Basal cell carcinoma often appears as a slightly transparent bump on the skin, though it can take other forms. Basal cell carcinoma occurs most often on areas of the skin that are exposed to the sun, such as your head and neck"
                        elif class_ind == 2:
                                info = "Benign lichenoid keratosis (BLK) usually presents as a solitary lesion that occurs predominantly on the trunk and upper extremities in middle-aged women. The pathogenesis of BLK is unclear; however, it has been suggested that BLK may be associated with the inflammatory stage of regressing solar lentigo (SL)1"
                        elif class_ind == 3:
                                info = "Dermatofibromas are small, noncancerous (benign) skin growths that can develop anywhere on the body but most often appear on the lower legs, upper arms or upper back. These nodules are common in adults but are rare in children. They can be pink, gray, red or brown in color and may change color over the years. They are firm and often feel like a stone under the skin. "
                        elif class_ind == 4:
                                info = "A melanocytic nevus (also known as nevocytic nevus, nevus-cell nevus and commonly as a mole) is a type of melanocytic tumor that contains nevus cells. Some sources equate the term mole with ‘melanocytic nevus’, but there are also sources that equate the term mole with any nevus form."
                        elif class_ind == 5:
                                info = "Pyogenic granulomas are skin growths that are small, round, and usually bloody red in color. They tend to bleed because they contain a large number of blood vessels. They’re also known as lobular capillary hemangioma or granuloma telangiectaticum."
                        elif class_ind == 6:
                                info = "Melanoma, the most serious type of skin cancer, develops in the cells (melanocytes) that produce melanin — the pigment that gives your skin its color. Melanoma can also form in your eyes and, rarely, inside your body, such as in your nose or throat. The exact cause of all melanomas isn't clear, but exposure to ultraviolet (UV) radiation from sunlight or tanning lamps and beds increases your risk of developing melanoma."
                        #print(info)
                        predict.append(info)
                        print(imagespath[i])
                        context = {"prediction":info , "predict": result , "img":imagespath[i]}
                                           
                        return render(request, 'prediction.html' , context)
                            