# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 09:34:22 2020

@author: Mesoco
"""


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from IPython.display import display
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import load_model

def filter_data():
    num_skipped = 0
    for folder_name in ("Cat", "Dog"):
        folder_path = os.path.join("./PetImages", folder_name)
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            try:
                fobj = open(fpath, "rb")
                is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
            finally:
                fobj.close()
            if not is_jfif:
                num_skipped += 1
                os.remove(fpath)
                
    print("Delete %d images" %num_skipped)
    
def generate_data():
    image_size = (180, 180)
    batch_size = 64
    
    # khởi tạo ImageDataGenerator
    data_generator = ImageDataGenerator(rescale=1/255, validation_split=0.2)
    
    generated_train_data = data_generator.flow_from_directory(
        "./PetImages",
        target_size = image_size,
        batch_size = batch_size,
        class_mode = "binary",
        subset = "training"
        
    )
    
    generated_validation_data = data_generator.flow_from_directory(
        "./PetImages",
        target_size = image_size,
        batch_size = batch_size,
        class_mode = "binary",
        subset = "validation",
    )
    
    return generated_train_data, generated_validation_data
    
def neural_net():
    model = Sequential()
    model.add(Conv2D(16, (3,3), activation='relu', input_shape=(180,180,3)))
    model.add(MaxPooling2D(2,2))
    model.add(Conv2D(32,(3,3), activation='relu'))
    model.add(MaxPooling2D(2,2))
    model.add(Flatten())
    model.add(Dense(512, activation='relu')) 
    model.add(Dense(1, activation='sigmoid'))               # xác suất nhỏ hơn 0.5 kết quả thuốc class 0 - mèo và lớn hơn 0.5 thuộc class 1 - chó
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer = opt, loss='binary_crossentropy', metrics=['acc'])
    
    return model

def predict_image(nameFile):
    img = keras.preprocessing.image.load_img(nameFile, target_size=(180,180))
    img_array = keras.preprocessing.image.img_to_array(img)
    #print(type(img_array[0][0][0]))
    plt.imshow(np.uint8(img_array))  # kiểu dữ liệu trong ảnh là float nên phải chuyển thành int trong range[0,255] hoặc float trong range[0,1]
    img_array = tf.expand_dims(img_array, axis=0)  # tạo thêm trục cho đúng kích thước keras yêu cầu (mở rộng chiều-tạo 1 hàng ở trục 0 tương ứng N=1)
    print(img_array.shape)
    model = load_model('./dogandcat.h5')
    predictions = model.predict(img_array)
    score = predictions[0]
    
    print('This image is %.2f percent dog and %.2f percent cat' %(100*score, 100*(1-score)))
              
if __name__ == '__main__':
    #filter_data()
    '''
    train_data, val_data = generate_data()
    
    model = neural_net()
    callback = tf.keras.callbacks.EarlyStopping(monitor='acc', patience= 2, restore_best_weights= True)
    model.fit_generator(train_data, validation_data= val_data, steps_per_epoch= 18739//64, epochs= 20, callbacks= [callback])
    
    model.save('dogandcat.h5')
    '''
    
    nameFile ='./PetImages/Dog/9917.jpg'
    predict_image(nameFile)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
