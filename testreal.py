'''Uses Keras to test a 2dconvlstm on VERITAS data.
Written by S.T. Spencer 8/7/2019'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib as mpl
mpl.use('Agg')
from keras.models import load_model
import numpy as np
import h5py
import keras
import os
import tempfile
import sys
from keras.utils import HDF5Matrix
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Conv2D, ConvLSTM2D, MaxPooling2D, BatchNormalization, Conv3D, GlobalAveragePooling3D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import AveragePooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers import Input, GaussianNoise
from keras.models import Model
from keras.layers import concatenate
from keras import backend as K
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.utils import plot_model
import matplotlib.pyplot as plt
import glob
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping
from keras import regularizers
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy.ma as ma
from matplotlib.pyplot import cm
from sklearn.preprocessing import scale
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy.ma as ma
from matplotlib.pyplot import cm
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from keras.metrics import binary_accuracy
from sklearn.metrics import roc_curve, auc
from net_utils import *

plt.ioff()

# Finds all the hdf5 files in a given directory
global realdata
realdata = sorted(glob.glob('/store/spencers/Data/Processed/*.hdf5'))
runname = 'vtest1'
runcode = 64080
hexmethod='axial_addressing'
modfile='/home/spencers/Models/vtest1model.hdf5' #Model file to use
evlist=[]

for file in realdata:
    try:
        h5file=h5py.File(file, 'r')
    except OSError:
        realdata.remove(file)
        continue
    events=h5file['id'][:].tolist()
    evlist=evlist+events
    h5file.close()

noev=len(evlist)
batchsize=50
no_steps=int(noev/float(batchsize))

global Trutharr
print(realdata,len(realdata))

print('No_events:',noev,'No_steps',no_steps)

# Define model architecture.
if hexmethod in ['axial_addressing','image_shifting']:
    inpshape=(None,27,27,1)
elif hexmethod in ['bicubic_interpolation','nearest_interpolation','oversampling','rebinning']:
    inpshape=(None,54,54,1)
else:
    print('Invalid Hexmethod')
    raise KeyboardInterrupt

# Test the network
print('Predicting')
model=load_model(modfile)

pred = model.predict_generator(
    generate_real_sequences(realdata,
        batchsize,hexmethod),
    verbose=0,
     use_multiprocessing=False,
    steps=no_steps)

np.save('/home/spencers/predictions/'+str(runcode)+'_'+runname+'_predictions_REAL.npy', pred)
