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
from keras.regularizers import l2
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Conv2D, ConvLSTM2D, MaxPooling2D, BatchNormalization, Conv3D, GlobalAveragePooling3D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import AveragePooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers import Input, GaussianNoise, Layer
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
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from net_utils import *
from keras import activations, initializers
from keras import callbacks, optimizers
import tqdm
import random
import time
import pandas as pd
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
from keras.regularizers import l2
import sklearn

K.clear_session()
plt.ioff()

# Finds all the hdf5 files in a given directory
global onlyfiles

onlyfiles = sorted(glob.glob('/store/spencers/Data/Crabrun2/*.hdf5'))
runname = 'crabrun2bayes'
hexmethod='oversampling'

global Trutharr
Trutharr = []
Train2=[]
truid=[]
print(onlyfiles,len(onlyfiles))

# Find true event classes for test data to construct confusion matrix.
for file in onlyfiles[120:160]:
    try:
        inputdata = h5py.File(file, 'r')
    except OSError:
        continue
    labelsarr = np.asarray(inputdata['isGamma'][:])
    idarr = np.asarray(inputdata['id'][:])
    for value in labelsarr:
        Trutharr.append(value)
    for value in idarr:
        truid.append(value)
    inputdata.close()

for file in onlyfiles[:120]:
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
lentrain=len(Train2)
lentruth=len(Trutharr)

dropout = 0.5
tau = 0.5  # obtained from BO
lengthscale = 1e-2
reg = lengthscale ** 2 * (1 - dropout) / (2. * lentrain * tau)

np.save('/home/spencers/truesim/truthvals_'+runname+'.npy',np.asarray(Trutharr))
np.save('/home/spencers/idsim/idvals_'+runname+'.npy',np.asarray(truid))

# Define model architecture.
if hexmethod in ['axial_addressing','image_shifting']:
    inpshape=(None,27,27,1)
elif hexmethod in ['bicubic_interpolation','nearest_interpolation','oversampling','rebinning']:
    inpshape=(None,54,54,1)
else:
    print('Invalid Hexmethod')
    raise KeyboardInterrupt

model = Sequential()
model.add(ConvLSTM2D(filters=30, kernel_size=(3, 3),
                     input_shape=inpshape,
                     padding='same', return_sequences=True,kernel_regularizer=keras.regularizers.l2(),dropout=0.5,recurrent_dropout=0.5))
model.add(BatchNormalization())

model.add(ConvLSTM2D(filters=30, kernel_size=(3, 3),
                     padding='same', return_sequences=True,dropout=0.5,recurrent_dropout=0.5,kernel_regularizer=keras.regularizers.l2()))
model.add(BatchNormalization())

model.add(GlobalAveragePooling3D())
model.add(Dense(20, activation='relu', kernel_regularizer= l2(reg)))
model.add(Dropout(0.5))
model.add(Dense(2,activation='softmax'))

opt = keras.optimizers.Adam()
#opt=keras.optimizers.Adadelta()
# Compile the model
model.compile(
    loss='binary_crossentropy',
    optimizer=opt,
    metrics=['binary_accuracy'])

'''early_stop = EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=10,
    verbose=1,
    mode='auto')'''

# Code for ensuring no contamination between training and test data.
print(model.summary())

plot_model(
    model,
    to_file='/home/spencers/Figures/'+runname+'_model.png',
    show_shapes=True)
# Train the network
#steps_per_epoch=lentrain/10.0,
#validation_steps=lentruth/10.0
history = model.fit_generator(
    generate_training_sequences(onlyfiles,
        10,
                                'Train',hexmethod),
    epochs=3,
    verbose=1,
    workers=0,
    use_multiprocessing=False,steps_per_epoch=10,
    shuffle=True,validation_data=generate_training_sequences(onlyfiles,10,'Valid',hexmethod),validation_steps=10
)

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

#steps=len(Trutharr)

predict_stochastic = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-1].output])

Yt_hat=[]
for i in tqdm.tqdm(range(10)): #No posterior samples
    dat=generate_training_sequences(onlyfiles,1,'Test',hexmethod)
    p0=[]
    for j in range(100): #No predict images
        p1=predict_stochastic([next(dat)[0],1])
        p0.append(p1)
    #p0=np.asarray(p0)
    Yt_hat.append(p0)

Yt_hat=np.asarray(Yt_hat)
print('yth',np.shape(Yt_hat))
np.save('bayespred',Yt_hat)

print('Evaluating')
#steps=len(Trutharr)/10.0
score = model.evaluate_generator(generate_training_sequences(onlyfiles,10,'Test',hexmethod),workers=0,use_multiprocessing=False,steps=10)
model.save('/home/spencers/Models/'+runname+'model.hdf5')

print('Test loss:', score[0])
print('Test accuracy:', score[1])


stoch_preds = pd.DataFrame()
print(np.shape(Yt_hat))
print("Calculating means etc...")
X_test=Trutharr[:100]
for i in range(len(X_test)):
    stoch_preds.set_value(index = i, col= 0, value = Yt_hat[:, 0, i, 0].mean())
    stoch_preds.set_value(index=i, col=1, value=Yt_hat[:, 0, i, 1].mean())

print(stoch_preds.to_string())
bayesian_predictions = stoch_preds.apply(lambda x: np.argmax(x), axis = 1)
print(bayesian_predictions.to_string())
raise KeyboardInterrupt
y_true = pd.Series([int(x)  for x in y_test])

confuse_mat_bayesian = pd.crosstab(y_true, bayesian_predictions)

print("Bayesian confusion matrix: \n {}".format(confuse_mat_bayesian))

def accuracy(confusion_matrix):
    acc = sum(np.diag(confusion_matrix))/sum(np.array(confusion_matrix)).sum()
    return acc

print("Bayesian accuracy: {}".format(accuracy(confuse_mat_bayesian)))

one_true = [1 if x == 1 else 0 for x in y_test]

print("Calculating precision/recall")

precision1_stoch, recall1_stoch, _ = precision_recall_curve(one_true, preds_df['mean_pred_1'],  pos_label=1)

average_precision1_stoch = average_precision_score(one_true, preds_df['mean_pred_1'])
print("Average Bayesian precision on Class 1: {}".format(average_precision1_stoch))
