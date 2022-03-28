# car_removal

## Installation libraries

```
pip install -U albumentations --no-binary qudida,albumentations
pip install -U git+https://github.com/albumentations-team/albumentations

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, add
from keras.layers import LeakyReLU, Activation, Conv2DTranspose, BatchNormalization, Add
from keras.layers import UpSampling2D, Conv2D, SeparableConv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from keras.applications import vgg19
from keras.models import load_model
import keras.backend as K

from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import albumentations as A

import torch
import torchvision
import torchvision.transforms as transforms
import math
import scipy

import shutil
import sys
import os
from tqdm import tqdm
import random
import time

from tensorflow.python.platform.tf_logging import set_verbosity, FATAL
```
