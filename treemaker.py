import numpy as np
from ROOT import TFile
from root_numpy import array2tree
import os

runname='vtest1_sim'
pred=np.load('/lustre/fs19/group/cta/users/sspencer/predictions/'+runname+'_predictions.npy')
ids=np.load('/lustre/fs19/group/cta/users/sspencer/idsim/idvals_'+runname+'.npy')
truth=np.load('/lustre/fs19/group/cta/users/sspencer/truesim/truthvals_'+runname+'.npy')
outfile='/lustre/fs19/group/cta/users/sspencer/trees/simtree_'+runname+'.root'
os.system('rm '+outfile)

roofile=TFile(outfile,'recreate')

outarr=np.core.records.fromarrays([pred,ids,truth],names='Prediction,id,truth')
outarr=array2tree(outarr,name='data_DL')
roofile.Write()
roofile.Close()
