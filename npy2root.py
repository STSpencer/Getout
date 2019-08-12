'''Script to take an existing DST file and add a DL_gammaness tree to it, plus add sig eff/bg rej hists'''


import ROOT
import numpy as np
import matplotlib.pyplot as plt
from rootpy.plotting import Hist
from root_numpy import array2hist,array2tree,array2root
from ROOT import TFile
from ROOT import gSystem, TFile, TTreeReader
import os

if gSystem.Load("$EVNDISPSYS/lib/libVAnaSum.so"):
    print ("Problem with evndisp")

runname='vtest1'
cutval=0.2
runfile='/lustre/fs19/group/cta/users/sspencer/ver/aux/64080.root'
predfile='/lustre/fs19/group/cta/users/sspencer/predictions/64080_vtest1_predictions_REAL.npy'

outfile='/lustre/fs19/group/cta/users/sspencer/ver/aux/64080_'+runname+'_DL.root'
sigfile='/lustre/fs19/group/cta/users/sspencer/confmatdata/'+runname+'_sigef.npy'
bgfile='/lustre/fs19/group/cta/users/sspencer/confmatdata/'+runname+'_bgrej.npy'

try:
    os.system('rm '+outfile) #Delete any existing room files with output name, stops issues with rewriting root files
except Exception:
    pass

os.system('cp '+runfile+' '+outfile) 
fpr=np.load(sigfile)
tpr=np.load(bgfile)
pred=np.load(predfile)-0.5
print(pred)
isGam=[]
print(np.shape(pred))
print(np.shape(pred)[0])
for i in np.arange(np.shape(pred)[0]):
    if pred[i]>cutval:
        isGam.append(1)
    else:
        isGam.append(0)

isGam=np.asarray(isGam)

roofile=TFile(outfile,'update')
pred=np.core.records.fromarrays([pred,isGam],names='dl_gammaness,dl_isGamma',formats='f8,bool')
pred=array2tree(pred,name='data_DL')
#pred.Scan()


roofile.Write()
roofile.Close()
#print(pred['dl_gammaness'])
