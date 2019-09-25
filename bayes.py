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
from sklearn.metrics import roc_curve, auc
from net_utils import *
from keras import activations, initializers
from keras import callbacks, optimizers

plt.ioff()

# Finds all the hdf5 files in a given directory
global onlyfiles
noise = 1.0

onlyfiles = sorted(glob.glob('/store/spencers/Data/Crabrun2/*.hdf5'))
runname = 'crabrun2bayes'
hexmethod='oversampling'

global Trutharr
Trutharr = []
Train2=[]
truid=[]
print(onlyfiles,len(onlyfiles))

def neg_log_likelihood(y_obs, y_pred, sigma=noise):
    dist = tf.distributions.Normal(loc=y_pred, scale=sigma)
    return K.sum(-dist.log_prob(y_obs))

def mixture_prior_params(sigma_1, sigma_2, pi, return_sigma=False):
    params = K.variable([sigma_1, sigma_2, pi], name='mixture_prior_params')
    sigma = np.sqrt(pi * sigma_1 ** 2 + (1 - pi) * sigma_2 ** 2)
    return params, sigma

def log_mixture_prior_prob(w):
    comp_1_dist = tf.distributions.Normal(0.0, prior_params[0])
    comp_2_dist = tf.distributions.Normal(0.0, prior_params[1])
    comp_1_weight = prior_params[2]    
    return K.log(comp_1_weight * comp_1_dist.prob(w) + (1 - comp_1_weight) * comp_2_dist.prob(w))    

# Mixture prior parameters shared across DenseVariational layer instances
prior_params, prior_sigma = mixture_prior_params(sigma_1=1.0, sigma_2=0.1, pi=0.2)

class DenseVariational(Layer):
    def __init__(self, output_dim, kl_loss_weight, activation=None, **kwargs):
        self.output_dim = output_dim
        self.kl_loss_weight = kl_loss_weight
        self.activation = activations.get(activation)
        super().__init__(**kwargs)

    def build(self, input_shape):  
        self._trainable_weights.append(prior_params) 

        self.kernel_mu = self.add_weight(name='kernel_mu', 
                                         shape=(input_shape[1], self.output_dim),
                                         initializer=initializers.normal(stddev=prior_sigma),
                                         trainable=True)
        self.bias_mu = self.add_weight(name='bias_mu', 
                                       shape=(self.output_dim,),
                                       initializer=initializers.normal(stddev=prior_sigma),
                                       trainable=True)
        self.kernel_rho = self.add_weight(name='kernel_rho', 
                                          shape=(input_shape[1], self.output_dim),
                                          initializer=initializers.constant(0.0),
                                          trainable=True)
        self.bias_rho = self.add_weight(name='bias_rho', 
                                        shape=(self.output_dim,),
                                        initializer=initializers.constant(0.0),
                                        trainable=True)
        super().build(input_shape)

    def call(self, x):
        kernel_sigma = tf.nn.softplus(self.kernel_rho)
        kernel = self.kernel_mu + kernel_sigma * tf.random_normal(self.kernel_mu.shape)

        bias_sigma = tf.nn.softplus(self.bias_rho)
        bias = self.bias_mu + bias_sigma * tf.random_normal(self.bias_mu.shape)
                
        self.add_loss(self.kl_loss(kernel, self.kernel_mu, kernel_sigma) + 
                      self.kl_loss(bias, self.bias_mu, bias_sigma))
        
        return self.activation(K.dot(x, kernel) + bias)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
    
    def kl_loss(self, w, mu, sigma):
        variational_dist = tf.distributions.Normal(mu, sigma)
        return kl_loss_weight * K.sum(variational_dist.log_prob(w) - log_mixture_prior_prob(w))




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
kl_loss_weight = 1.0 / (len(Train2)/10.0)


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
model.add(Dense(20,activation='relu'))
model.add(DenseVariational(20,kl_loss_weight=kl_loss_weight,activation='relu'))
model.add(DenseVariational(20,kl_loss_weight=kl_loss_weight,activation='relu'))
model.add(DenseVariational(2,kl_loss_weight=kl_loss_weight,activation='softmax'))

opt = keras.optimizers.Adam(lr=0.03)

# Compile the model
model.compile(
    loss=neg_log_likelihood,
    optimizer=opt,
    metrics=['mse'])

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
history = model.fit_generator(
    generate_training_sequences(onlyfiles,
        10,
                                'Train',hexmethod),
    steps_per_epoch=lentrain/10.0,
    epochs=1,
    verbose=1,
    workers=0,
    use_multiprocessing=False,
    shuffle=True,validation_data=generate_training_sequences(onlyfiles,10,'Valid',hexmethod),validation_steps=lentruth/10.0)
'''
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
'''
# Test the network
print('Predicting')
pred = model.predict_generator(
    generate_training_sequences(onlyfiles,
        1,
        'Test',hexmethod),
    verbose=0,workers=0,
     use_multiprocessing=False,
    steps=len(Trutharr))
np.save('/home/spencers/predictions/'+runname+'_predictions.npy', pred)
print(pred[:100],np.shape(pred))
y_preds=np.concatenate(pred,axis=1)
y_mean=np.mean(y_preds,axis=1)
y_sigma=np.std(y_preds,axis=1)
X_test=np.linspace(0,1,len(pred)).reshape(-1,1)

plt.plot(X_test, y_mean, 'r-', label='Predictive mean');
plt.fill_between(X_test.ravel(), 
                 y_mean + 2 * y_sigma, 
                 y_mean - 2 * y_sigma, 
                 alpha=0.5, label='Epistemic uncertainty')
plt.title('Prediction')
plt.savefig('bayes1.png')

'''print('Evaluating')

score = model.evaluate_generator(generate_training_sequences(onlyfiles,10,'Test',hexmethod),workers=0,use_multiprocessing=False,steps=len(Trutharr)/10.0)
model.save('/home/spencers/Models/'+runname+'model.hdf5')

print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Plot confusion matrix


print(get_confusion_matrix_one_hot(runname,pred, Trutharr))
fig=plt.figure()
Trutharr=np.asarray(Trutharr)
noev=min([len(Trutharr),len(pred)])
pred=pred[:noev]
Trutharr=Trutharr[:noev]
x1=np.where(Trutharr==0)
x2=np.where(Trutharr==1)
p2=[]
print(pred,np.shape(pred))

for i in np.arange(np.shape(pred)[0]):
    score=np.argmax(pred[i])
    if score==0:
        s2=1-pred[i][0]
    elif score==1:
        s2=pred[i][1]
    p2.append(s2)


p2=np.asarray(p2)
np.save('/home/spencers/predictions/'+runname+'_predictions.npy', p2)
x1=x1[0]
x2=x2[0]

plt.hist(p2[x1],10,label='True Hadron',alpha=0.5,density=False)
plt.hist(p2[x2],10,label='True Gamma',alpha=0.5,density=False)
plt.xlabel('isGamma Score')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('/home/spencers/Figures/'+runname+'_hist.png')
cutval=0.1
print('No_gamma',len(np.where(p2>=cutval)[0]))
print('No_bg',len(np.where(p2<cutval)[0]))
'''
