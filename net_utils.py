from __future__ import absolute_import, division, print_function

import matplotlib as mpl
import numpy as np
import keras
import os, tempfile, sys, glob, h5py
from keras.utils import HDF5Matrix
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Input, GaussianNoise, concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import AveragePooling2D
from keras.layers.core import Activation, Dropout
from keras import backend as K
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.utils import plot_model
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping
from keras import regularizers
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy.ma as ma
from  matplotlib.pyplot import cm
from sklearn.preprocessing import scale
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from keras.metrics import binary_accuracy
#from ctapipe.image import tailcuts_clean
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import cycle

def get_confusion_matrix_one_hot(runname,model_results, truth):
    '''model_results and truth should be for one-hot format, i.e, have >= 2 columns,
    where truth is 0/1, and max along each row of model_results is model result
    '''
    mr=[]
    mr2=[]
    mr3=[]
    print(model_results,truth)
    for x in model_results:
        mr.append(np.argmax(x))
        mr2.append(x)
    mr3 = label_binarize(mr, classes=[0, 1, 2])
    no_ev=min(len(mr),len(truth))
    print(no_ev)
    model_results=np.asarray(mr)[:no_ev]
    truth=np.asarray(truth)[:no_ev]
    print(np.shape(model_results),np.shape(truth))
    mr2=mr2[:no_ev]
    mr3=mr3[:no_ev]
    cm=confusion_matrix(y_target=truth,y_predicted=np.rint(np.squeeze(model_results)),binary=False)
    fig,ax=plot_confusion_matrix(conf_mat=cm,figsize=(5,5))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('/home/spencers/Figures/'+runname+'confmat.png')
    lw=2
    n_classes=2
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    t2 = label_binarize(truth, classes=[0, 1])
    print(mr2[:100])
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(t2[:, i], np.asarray(mr2)[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    fpr["micro"], tpr["micro"], _ = roc_curve(t2.ravel(), mr3.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='Micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='Macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    plt.legend(loc="lower right")
    plt.savefig('/home/spencers/Figures/'+runname+'_roc.png')
    np.save('/home/spencers/confmatdata/'+runname+'_fp.npy',fpr)
    np.save('/home/spencers/confmatdata/'+runname+'_tp.npy',tpr)
    return cm

def generate_training_sequences2d(onlyfiles,batch_size, batchflag, shilonflag=True):
    """ Generates training/test sequences on demand
    """

    nofiles = 0
    i = 0  # No. events loaded in total

    if batchflag == 'Train':
        filelist = onlyfiles[:160]
        global trainevents
        global train2
        for file in filelist:
            inputdata = h5py.File(file, 'r')
            trainevents = trainevents + inputdata['event_id'][:].tolist()
            train2 = train2 + inputdata['id'][:].tolist()
            inputdata.close()

    elif batchflag == 'Test':
        filelist = onlyfiles[160:320]
        global testevents
        global test2
        for file in filelist:
            inputdata = h5py.File(file, 'r')
            testevents = testevents + inputdata['event_id'][:].tolist()
            test2 = test2 + inputdata['id'][:].tolist()
            inputdata.close()
    else:
        print('Error: Invalid batchflag')
        raise KeyboardInterrupt

    while True:
        for file in filelist:
            inputdata = h5py.File(file, 'r')
            chargearr = np.asarray(inputdata['squared_training'][:, 0, :, :])
            chargearr = chargearr[:,8:40, 8:40]
            labelsarr = np.asarray(inputdata['labels'][:,0])
            valilocs = np.where(labelsarr!=-1)[0]
            labelsarr = labelsarr[valilocs]
            chargearr = chargearr[valilocs,:,:]
            idarr = np.asarray(inputdata['id'][:])
            nofiles = nofiles + 1
            inputdata.close()
            chargearr = np.reshape(chargearr, (np.shape(chargearr)[0], 32, 32, 1))
            
            training_sample_count = len(chargearr)
            batches = int(training_sample_count / batch_size)
            remainder_samples = training_sample_count % batch_size
            i = i + 1000
            countarr = np.arange(0, len(labelsarr))
            ta2 = np.zeros((training_sample_count, 32, 32, 1))
            
            ta2[:, :, :, 0] = chargearr[:, :, :, 0]

            trainarr = ta2
            trainarr = (trainarr-np.amin(trainarr,axis=0))/(np.amax(trainarr,axis=0)-np.amin(trainarr,axis=0))

            if remainder_samples:
                batches = batches + 1

            # generate batches of samples
            for idx in list(range(0, batches)):
                if idx == batches - 1:
                    batch_idxs = countarr[idx * batch_size:]
                else:
                    batch_idxs = countarr[idx *
                                          batch_size:idx *
                                          batch_size +
                                          batch_size]
                X = trainarr[batch_idxs]
                X = np.nan_to_num(X)
                Y = keras.utils.to_categorical(
                    labelsarr[batch_idxs], num_classes=2)
                yield (np.array(X), np.array(Y))

def generate_training_sequences(onlyfiles,batch_size, batchflag,hexmethod):
    """ Generates training/test sequences on demand
    """

    nofiles = 0
    i = 0  # No. events loaded in total

    if batchflag == 'Train':
        filelist = onlyfiles[0:4]
        print('train', filelist)
        global trainevents
        global train2
        for file in filelist:
            inputdata = h5py.File(file, 'r')
            trainevents = trainevents + inputdata['isGamma'][:].tolist()
            train2 = train2 + inputdata['id'][:].tolist()
            inputdata.close()

    elif batchflag == 'Test':
        filelist = onlyfiles[4:6]
        print('test', filelist)
        global testevents
        global test2
        for file in filelist:
            inputdata = h5py.File(file, 'r')
            testevents = testevents + inputdata['isGamma'][:].tolist()
            test2 = test2 + inputdata['id'][:].tolist()
            inputdata.close()

    elif batchflag == 'Valid':
        '''filelist = onlyfiles[30:40]
        print('valid', filelist)
        global validevents
        global valid2
        for file in filelist:
            inputdata = h5py.File(file, 'r')
            validevents = validevents + inputdata['event_id'][:].tolist()
            valid2 = valid2 + inputdata['id'][:].tolist()
            inputdata.close()'''
    else:
        print('Error: Invalid batchflag')
        raise KeyboardInterrupt
    
    while True:
        for file in filelist:
            inputdata = h5py.File(file, 'r')
            trainarr = np.asarray(inputdata[hexmethod][:, :, :, :])
            labelsarr = np.asarray(inputdata['isGamma'][:])
            idarr = np.asarray(inputdata['id'][:])
            nofiles = nofiles + 1
            inputdata.close()
            notrigs=np.shape(trainarr)[0]
            
            for x in np.arange(np.shape(trainarr)[0]):
                chargevals = []
                for y in np.arange(4):
                    chargevals.append(np.sum(trainarr[x,y,:,:]))

                chargevals = np.argsort(chargevals)
                chargevals = np.flip(chargevals,axis=0) #Flip to descending order.
                trainarr[x, :, :, :] = trainarr[x, chargevals, :, :]
                            
            training_sample_count = len(trainarr)
            batches = int(training_sample_count / batch_size)
            remainder_samples = training_sample_count % batch_size
            i = i + 1000
            countarr = np.arange(0, len(labelsarr))

            trainarr = (trainarr-np.amin(trainarr,axis=0))/(np.amax(trainarr,axis=0)-np.amin(trainarr,axis=0))
            if remainder_samples:
                batches = batches + 1

            # generate batches of samples
            for idx in list(range(0, batches)):
                if idx == batches - 1:
                    batch_idxs = countarr[idx * batch_size:]
                else:
                    batch_idxs = countarr[idx *
                                          batch_size:idx *
                                          batch_size +
                                          batch_size]
                X = trainarr[batch_idxs]
                X = np.nan_to_num(X)
                Y = keras.utils.to_categorical(
                    labelsarr[batch_idxs], num_classes=2)
                yield (np.array(X), np.array(Y))
