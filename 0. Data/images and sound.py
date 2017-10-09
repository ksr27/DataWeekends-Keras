# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 16:51:00 2017

@author: Stefan Draghici
"""

# IMAGES
from PIL import Image
import numpy as np

# import the image
img=Image.open('iss.jpg')

# convert the image into a matrix of bytes
img_array=np.asarray(img)
img_array.shape

# flatten the matrix to an array
img_array.ravel().shape

# SOUND
from scipy.io import wavfile as wf

# load the sound file and split the frequency from the actual sound bytes
rate, snd=wf.read(filename='sms.wav')