#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 18:45:05 2020

@author: chetanhirapara
"""

import numpy as np
# from keras.models import load_model
from keras.preprocessing import image

# from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

class classification:
    def __init__(self,filename):
        self.filename =filename

    def prediction(self):
        # load model
        model = load_model('model.h5')

        # summarize model
        #model.summary()
        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (128, 128))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)

        classes = ['black_dress', 'black_pants', 'black_shirt', 'black_shoes', 'black_shorts', 'blue_dress',
                   'blue_pants', 'blue_shirt', 'blue_shoes', 'blue_shorts', 'brown_pants', 'brown_shoes',
                   'brown_shorts', 'green_pants', 'green_shirt', 'green_shoes', 'green_shorts', 'red_dress',
                   'red_pants', 'red_shoes', 'white_dress', 'white_pants', 'white_shoes', 'white_shorts']

        pred = classes[np.argmax(result)]
        return [{ "image" : pred}]
