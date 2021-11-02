import numpy as np
import matplotlib.pyplot as plt
import glob
import h5py
from matplotlib import ticker, cm

pred=np.load('bayespred2.npy')

global onlyfiles
onlyfiles = sorted(glob.glob('/store/spencers/Data/Crabrun2/*.hdf5'))

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
Trutharr=np.asarray(Trutharr)
print(Trutharr,np.shape(Trutharr))

p2=np.zeros((np.shape(pred)[0],np.shape(pred)[1]))
print(np.shape(p2))
print(pred,np.shape(pred))
for j in np.arange(np.shape(pred)[0]):
    for i in np.arange(np.shape(pred)[1]):
        score=np.argmax(pred[j][i])
        if score==0:
            s2=1-pred[j][i][0]
        elif score==1:
            s2=pred[j][i][1]
        p2[j][i]=s2


print(np.shape(p2))

X=np.arange(20)
Y=np.arange(len(Trutharr))
print(np.shape(X),np.shape(Y))
fig, ax = plt.subplots()
cs = ax.contourf(Y, X, p2)
plt.xlabel('Event')
plt.ylabel('Test Iteration')
cbar = fig.colorbar(cs)
plt.savefig('bayes2.png')
#plt.show()
print(X)