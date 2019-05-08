def imports():
    import os
    import pandas as pd
    import numpy as np
    import argparse
    import random
    import sklearn
    import sklearn.metrics as metrics
    from sklearn.metrics import confusion_matrix
    #set the matplotlib backend so plots can be saved in the background
    #necessary since running on Google Cloud  
    import matplotlib
    %matplotlib inline
    matplotlib.use('Agg') 
    import matplotlib.pyplot as plt
    
    #Utilize GPUs
    import tensorflow as tf
    #Setup Keras 
    from keras.models import Sequential, Model
    from keras.layers.convolutional import Conv2D, MaxPooling2D
    from keras.layers.core import Activation, Flatten, Dense
    from keras.layers import BatchNormalization, Dropout, LeakyReLU
    from keras.optimizers import Adam, SGD, Adagrad
    from keras import backend as K 
    K.tensorflow_backend._get_available_gpus()
    
    from keras.preprocessing.image import ImageDataGenerator    
    from keras.callbacks import History
    
    #Setup VGG16
    from keras.applications import vgg16
    from keras.preprocessing import image
    from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions

    

