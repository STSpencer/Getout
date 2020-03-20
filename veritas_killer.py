import numpy as np
import h5py
import keras
import matplotlib.pyplot as plt
import sklearn
from keras.models import load_model
import glob
from net_utils import *

def fourblockdead(images,hexmethod):
    if hexmethod in ['axial_addressing','image_shifting']:
        inpran=27
    elif hexmethod in ['bicubic_interpolation','nearest_interpolation','oversamplin\
g','rebinning']:
        inpran=54
    else:
        print('Invalid Hexmethod')
        raise KeyboardInterrupt

    for x in np.arange(np.shape(images)[0]):
        ix=np.random.randint(1,inpran-1)
        iy=np.random.randint(1,inpran-1)

        for y in np.arange(4):
            images[x,y,ix-1:ix,iy-1:iy]=0
    return images

def generate_corrupt(onlyfiles,batch_size, batchflag,hexmethod):
    """ Generates training/test sequences on demand
    """

    nofiles = 0
    i = 0  # No. events loaded in total
    testevents=[]
    trainevents=[]
    train2=[]
    test2=[]
    if batchflag == 'Train':
        filelist = onlyfiles[:120]
        print('train', filelist)
        for file in filelist:
            try:
                inputdata = h5py.File(file, 'r')
            except OSError:
                continue
            trainevents = trainevents + inputdata['isGamma'][:].tolist()
            train2 = train2 + inputdata['id'][:].tolist()
            inputdata.close()

    elif batchflag == 'Test':
        filelist = onlyfiles[120:160]
        print('test', filelist)
        for file in filelist:
            try:
                inputdata = h5py.File(file, 'r')
            except OSError:
                continue
            testevents = testevents + inputdata['isGamma'][:].tolist()
            test2 = test2 + inputdata['id'][:].tolist()
            inputdata.close()

    elif batchflag == 'Valid':
        filelist = onlyfiles[160:]
        print('valid', filelist)
        global validevents
        global valid2
        for file in filelist:
            inputdata = h5py.File(file, 'r')
            validevents = validevents + inputdata['isGamma'][:].tolist()
            valid2 = valid2 + inputdata['id'][:].tolist()
            inputdata.close()
    else:
        print('Error: Invalid batchflag')
        raise KeyboardInterrupt
    
    while True:
        for file in filelist:
            try:
                inputdata = h5py.File(file, 'r')
            except OSError:
                continue
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

#            trainarr = (trainarr-np.amin(trainarr,axis=0))/(np.amax(trainarr,axis=0)-np.amin(trainarr,axis=0))
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
                X = fourblockdead(X,hexmethod)
                Y = keras.utils.to_categorical(
                    labelsarr[batch_idxs], num_classes=2)
                yield (np.array(X), np.array(Y))

if __name__=="__main__":
    runname = 'crabrun2opt4_kill'
    runcode = 64080
    hexmethod='oversampling'
    modfile='/home/spencers/Models/crabrun2opt4model.hdf5'
    onlyfiles = sorted(glob.glob('/store/spencers/Data/Crabrun2/*.hdf5'))
    batchsize=50
    
    Trutharr=[]
    truid=[]
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
    noev=int(len(truid)/float(batchsize))
    
    # Test the network
    print('Predicting')
    model=load_model(modfile)
    g2=generate_corrupt(onlyfiles,batchsize,'Test',hexmethod)
    pred = model.predict_generator(g2,
                                   verbose=1,workers=0,
                                   use_multiprocessing=False,
                                   steps=noev)

    print(get_confusion_matrix_one_hot(runname,pred, Trutharr))
