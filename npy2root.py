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
#raise KeyboardInterrupt
fpr=np.load(sigfile)
tpr=np.load(bgfile)
pred=np.load(predfile)

pred=np.expand_dims(pred,axis=0)
roofile=TFile(outfile,'update')
pred=np.core.records.fromarrays(pred,names='dl_gammaness')
pred=array2tree(pred,name='data_DL')
#pred.Scan()

h1=Hist(len(fpr),0,len(fpr),title='Signal Efficiency')
h2=Hist(len(tpr),0,len(tpr),title='Background Rejection')
_=array2hist(fpr,h1)
_=array2hist(tpr,h2)

roofile.Write()
roofile.Close()
#print(pred['dl_gammaness'])
