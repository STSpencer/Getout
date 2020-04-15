import sys 
sys.argv.append( '-b-' )
import ROOT
import numpy as np
from root_numpy import tree2array
from ROOT import gSystem, TFile, TTreeReader
from root_numpy import tree2array,root2array,list_trees,list_branches
import root_numpy
import matplotlib.pyplot as plt
from ctapipe_hillas import hillas_parameters
import h5py

ROOT.ROOT.EnableImplicitMT()
ROOT.gROOT.SetBatch(True)
if gSystem.Load("$EVNDISPSYS/lib/libVAnaSum.so"):
    print("Problem loading EventDisplay libraries - please check this before proceeding")

protonfile='/lustre/fs19/group/cta/users/sspencer/ver/aux/mcprotoncrab2.dst.root' #This is a test tycho run just to use as a placeholder until we get some MC
cam = 'VERITAS'
runcode='Crabrun2_hillas'
runname='/lustre/fs19/group/cta/users/sspencer/Crabrun2/'+runcode+'_'
nochannels=499
nofiles=200

f=ROOT.TFile.Open(protonfile,'read')
mytree=f.Get("dst")
noprotons=len(tree2array(mytree,branches=['eventNumber']))
f.Close()

notrigs=noprotons
print('Number of triggers',notrigs)

def extract_hillas(i,imarr,tel):
    '''Extracts VERITAS images from root files and performs hexagonal image manipulation.'''
    #print(imarr[i][0][tel][:499])
    try:
        image=imarr[i][0][tel][:499]
    except IndexError:
        return None
    print(np.min(image))
    if np.min(image)<0.0:
        image=image+abs(np.min(image))
    params=hillas_parameters(image)
    return params

trigsperfile=int(float(notrigs)/float(nofiles))

for runno in np.arange(nofiles):
    to_hdf={'id':[],'isGamma':[],'oversampling':[],'rebinning':[],'nearest_interpolation':[],'bilinear_interpolation':[],'bicubic_interpolation':[],'image_shifting':[],'axial_addressing':[]}
    startev=runno*trigsperfile
    stopev=(runno+1)*trigsperfile
    print('Processing Protons')

    f=ROOT.TFile.Open(protonfile,'read')
    mytree=f.Get("dst")
    print('Proton DST Retrieved')
    imarr=tree2array(mytree,branches=['sum'],start=startev,stop=stopev)
    #deadarr=tree2array(mytree,branches=['dead'],start=startev,stop=stopev)
    evarr=tree2array(mytree,branches=['eventNumber'],start=startev,stop=stopev)
    
    for i in np.arange(trigsperfile):
        
        to_hdf['id'].append(trigsperfile*runno+evarr[i][0])
        to_hdf['isGamma'].append(0) #0 is proton, 1 is gamma

        h1=extract_hillas(i,imarr,0) #Should set deadarr to None when MC comes through
        h2=extract_hillas(i,imarr,1)
        h3=extract_hillas(i,imarr,2)
        h4=extract_hillas(i,imarr,3)
        if h1 is not None:
            print(h1,h2,h3,h4)

            f.Close()
            raise KeyboardInterrupt
        else:
            continue
    f.Close()
    raise KeyboardInterrupt
