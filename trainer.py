'''Uses Keras to train and test a 2dconvlstm on parameterized VERITAS data.
Written by S.T. Spencer 27/6/2019'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib as mpl
mpl.use('Agg')
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
global onlyfiles
onlyfiles = sorted(glob.glob('/store/spencers/Data/Processed/*.hdf5'))
runname = 'vtest1'
hexmethod='axial_addressing'

global Trutharr
Trutharr = []
Train2=[]
print(onlyfiles,len(onlyfiles))

# Find true event classes for test data to construct confusion matrix.
for file in onlyfiles[7:8]:
    print(file)
    try:
        inputdata = h5py.File(file, 'r')
    except OSError:
        continue
    for key in inputdata.keys():
        print(key,np.shape(inputdata[key]))
    labelsarr = np.asarray(inputdata['isGamma'][:])
    for value in labelsarr:
        Trutharr.append(value)
    inputdata.close()

for file in onlyfiles[1:6]:
    print(file)
    try:
        inputdata = h5py.File(file, 'r')
    except OSError:
        continue
    labelsarr = np.asarray(inputdata['isGamma'][:])
    for value in labelsarr:
        Train2.append(value)
    inputdata.close()

print('lentruth', len(Trutharr))
print('lentrain',len(Train2))


# Define model architecture.
if hexmethod in ['axial_addressing','image_shifting']:
    inpshape=(None,27,27,1)
elif hexmethod in ['bicubic_interpolation','nearest_interpolation','oversampling','rebinning']:
    inpshape=(None,54,54,1)
else:
    print('Invalid Hexmethod')
    raise KeyboardInterrupt

model = Sequential()
model.add(ConvLSTM2D(filters=20, kernel_size=(3, 3),
                     input_shape=inpshape,
                     padding='same', return_sequences=True,recurrent_regularizer=keras.regularizers.l2()))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(ConvLSTM2D(filters=20, kernel_size=(3, 3),
                    padding='same', return_sequences=True,recurrent_regularizer=keras.regularizers.l2()))
model.add(Dropout(0.5))
#model.add(BatchNormalization())
#model.add(ConvLSTM2D(filters=30, kernel_size=(3, 3),
                     #padding='same', return_sequences=True))
#model.add(Dropout(0.5))
#model.add(BatchNormalization())

#model.add(ConvLSTM2D(filters=30, kernel_size=(3, 3),
                     #padding='same', return_sequences=True))
#model.add(Dropout(0.5))
#model.add(BatchNormalization())

#model.add(ConvLSTM2D(filters=30, kernel_size=(3, 3),
                     #padding='same', return_sequences=True))
#model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(GlobalAveragePooling3D())
model.add(Dense(2, activation='sigmoid'))
opt = keras.optimizers.Adadelta()

# Compile the model
model.compile(
    loss='binary_crossentropy',
    optimizer=opt,
    metrics=['binary_accuracy'])

early_stop = EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=10,
    verbose=1,
    mode='auto')

# Code for ensuring no contamination between training and test data.
print(model.summary())

plot_model(
    model,
    to_file='/home/spencers/Figures/'+runname+'_model.png',
    show_shapes=True)

# Train the network
history = model.fit_generator(
    generate_training_sequences(onlyfiles,
        50,
                                'Train',hexmethod),
    steps_per_epoch=235,
    epochs=10,
    verbose=1,
    use_multiprocessing=False,
    shuffle=False,validation_data=generate_training_sequences(onlyfiles,50,'Valid',hexmethod),validation_steps=78)

# Plot training accuracy/loss.
fig = plt.figure()
plt.subplot(2, 1, 1)
print(history.history)
plt.plot(history.history['binary_accuracy'], label='Train')
plt.plot(history.history['val_binary_accuracy'],label='Validation')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')

plt.subplot(2, 1, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'],label='Validation')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')

plt.tight_layout()

plt.savefig('/home/spencers/Figures/'+runname+'trainlog.png')

# Test the network
print('Predicting')
pred = model.predict_generator(
    generate_training_sequences(onlyfiles,
        50,
        'Test',hexmethod),
    verbose=0,
     use_multiprocessing=False,
    steps=78)
np.save('/home/spencers/predictions/'+runname+'_predictions.npy', pred)

print('Evaluating')

score = model.evaluate_generator(
    generate_training_sequences(onlyfiles,
        50,'Test',hexmethod),
    use_multiprocessing=False,
    steps=156)
model.save('/home/spencers/Models/'+runname+'model.hdf5')

print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Plot confusion matrix


print(get_confusion_matrix_one_hot(runname,pred, Trutharr))
