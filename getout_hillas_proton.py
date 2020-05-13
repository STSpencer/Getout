import sys 
sys.argv.append( '-b-' )
import ROOT
import numpy as np
from root_numpy import tree2array
from ROOT import gSystem, TFile, TTreeReader
from root_numpy import tree2array,root2array,list_trees,list_branches
import root_numpy
from ctapipe_hillas import hillas_parameters
import h5py
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def dedead(imarr,tel,pixel):
    '''Attempts to correct for dead pixels by setting them equal to the mean of the 1st 2nd or 3rd nearest pair of non-zero pixels'''
    try:
        if imarr[i][0][tel][pixel+1] !=0.0 and imarr[i][0][tel][pixel-1] !=0.0:
            imarr[i][0][tel][pixel]=(imarr[i][0][tel][pixel+1]+imarr[i][0][tel][pixel-1])/2.0
        elif imarr[i][0][tel][pixel+2] !=0.0 and imarr[i][0][tel][pixel-2] !=0.0:
            imarr[i][0][tel][pixel]=(imarr[i][0][tel][pixel+2]+imarr[i][0][tel][pixel-2])/2.0
        elif imarr[i][0][tel][pixel+3] !=0.0 and imarr[i][0][tel][pixel-3] !=0.0:
            imarr[i][0][tel][pixel]=(imarr[i][0][tel][pixel+3]+imarr[i][0][tel][pixel-3])/2.0
        else:
            pass
            #print('Dead pixel not corrected: '+str(pixel))
    except IndexError:
        print('Index error')
    return imarr

ROOT.ROOT.EnableImplicitMT()
ROOT.gROOT.SetBatch(True)
if gSystem.Load("$EVNDISPSYS/lib/libVAnaSum.so"):
    print("Problem loading EventDisplay libraries - please check this before proceeding")

#protonfile='/lustre/fs19/group/cta/users/sspencer/ver/aux/mcgammacrab.dst.root' #This is a test tycho run just to use as a placeholder until we get some MC
protonfile='/lustre/fs19/group/cta/users/sspencer/ver/eventdisplay_output/64080.dst.root'
cam = 'VERITAS'
particle='Real'
nochannels=499
nofiles=10
Realdata=True

f=ROOT.TFile.Open(protonfile,'read')
mytree=f.Get("dst")
noprotons=len(tree2array(mytree,branches=['eventNumber']))
f.Close()

notrigs=noprotons

print('Number of triggers',notrigs)

def extract_hillas(i,imarr,tel,Realdata,deadvals):
    '''Extracts VERITAS images from root files and performs hexagonal image manipulation.'''
    #print(imarr[i][0][tel][:499])
    try:
        if Realdata==True:
            deadvals=deadarr[i][0][tel][:499]
            deadvals=np.expand_dims(deadvals,1)
            deadlist=np.where(deadvals!=0)[0]
            for k in deadlist:
                imarr=dedead(imarr,tel,k)
            image=imarr[i][0][tel][:499]
        else:
            image=imarr[i][0][tel][:499]

    except IndexError:
        return None

    if np.min(image)<0.0:
        image=image+abs(np.min(image))
    params=hillas_parameters(image)
    return params

trigsperfile=int(float(notrigs)/float(nofiles))
t1l=[]
t1w=[]
t2l=[]
t2w=[]
t3l=[]
t3w=[]
t4l=[]
t4w=[]
t1p=[]
t1s=[]
t1k=[]
t2p=[]
t2s=[]
t2k=[]
t3p=[]
t3s=[]
t3k=[]
t4p=[]
t4s=[]
t4k=[]

t1i=[]
t2i=[]
t3i=[]
t4i=[]

for runno in np.arange(nofiles):
    to_hdf={'id':[],'isGamma':[],'oversampling':[],'rebinning':[],'nearest_interpolation':[],'bilinear_interpolation':[],'bicubic_interpolation':[],'image_shifting':[],'axial_addressing':[]}
    startev=runno*trigsperfile
    if runno==nofiles:
        stopev=notrigs
    else:
        stopev=(runno+1)*trigsperfile
    print('Processing Protons')

    f=ROOT.TFile.Open(protonfile,'read')
    mytree=f.Get("dst")
    print('Proton DST Retrieved')
    imarr=tree2array(mytree,branches=['sum'],start=startev,stop=stopev)
    if Realdata==True:
        deadarr=tree2array(mytree,branches=['dead'],start=startev,stop=stopev)
    else:
        deadarr=None
    evarr=tree2array(mytree,branches=['eventNumber'],start=startev,stop=stopev)
    
    for i in np.arange(trigsperfile):
        

        h1=extract_hillas(i,imarr,0,Realdata,deadarr) #Should set deadarr to None when MC comes through
        h2=extract_hillas(i,imarr,1,Realdata,deadarr)
        h3=extract_hillas(i,imarr,2,Realdata,deadarr)
        h4=extract_hillas(i,imarr,3,Realdata,deadarr)
        if h1 is not None:
            t1i.append(h1[4][0])
            t2i.append(h2[4][0])
            t3i.append(h3[4][0])
            t4i.append(h4[4][0])
            t1l.append(h1[5][0].value)
            t1w.append(h1[6][0].value)
            t1p.append(h1[7][0].value)
            t1s.append(h1[8][0])
            t1k.append(h1[9][0])
            t2l.append(h2[5][0].value)
            t2w.append(h2[6][0].value)
            t2p.append(h2[7][0].value)
            t2s.append(h2[8][0])
            t2k.append(h2[9][0])
            t3l.append(h3[5][0].value)
            t3w.append(h3[6][0].value)
            t3p.append(h3[7][0].value)
            t3s.append(h3[8][0])
            t3k.append(h3[9][0])
            t4l.append(h4[5][0].value)
            t4w.append(h4[6][0].value)
            t4p.append(h4[7][0].value)
            t4s.append(h4[8][0])
            t4k.append(h4[9][0])
        else:
            continue
    f.Close()

np.save('../'+particle+'_t1i.npy',np.asarray(t1i))
np.save('../'+particle+'_t2i.npy',np.asarray(t2i))
np.save('../'+particle+'_t3i.npy',np.asarray(t3i))
np.save('../'+particle+'_t4i.npy',np.asarray(t4i))
np.save('../'+particle+'_t1w.npy',np.asarray(t1w))
np.save('../'+particle+'_t2w.npy',np.asarray(t2w))
np.save('../'+particle+'_t3w.npy',np.asarray(t3w))
np.save('../'+particle+'_t4w.npy',np.asarray(t4w))

fig=plt.figure()
plt.title(particle+' Hillas Length')
print(t1l)
plt.hist(t1l,label='Telescope 1',alpha=0.3)
plt.hist(t2l,label='Telescope 2',alpha=0.3)
plt.hist(t3l,label='Telescope 3',alpha=0.3)
plt.hist(t4l,label='Telescope 4',alpha=0.3)
plt.legend()
plt.xlabel('Value (m)')
plt.ylabel('Frequency (counts)')
plt.savefig('../Figures/'+particle+'_hl.png')
fig=plt.figure()
plt.title(particle+' Hillas Width')
plt.hist(t1w,label='Telescope 1',bins=200,alpha=0.1,range=(0.16,0.185))
plt.hist(t2w,label='Telescope 2',bins=200,alpha=0.1,range=(0.16,0.185))
plt.hist(t3w,label='Telescope 3',bins=200,alpha=0.1,range=(0.16,0.185))
plt.hist(t4w,label='Telescope 4',bins=200,alpha=0.1,range=(0.16,0.185))
plt.legend()
plt.xlabel('Value (m)')
plt.ylabel('Frequency (counts)')
plt.savefig('../Figures/'+particle+'_hw.png')
fig=plt.figure()
plt.title(particle+' Hillas Psi')
plt.hist(t1p,label='Telescope 1',alpha=0.3)
plt.hist(t2p,label='Telescope 2',alpha=0.3)
plt.hist(t3p,label='Telescope 3',alpha=0.3)
plt.hist(t4p,label='Telescope 4',alpha=0.3)
plt.legend()
plt.xlabel('Value (rad)')
plt.ylabel('Frequency (counts)')
plt.savefig('../Figures/'+particle+'_hp.png')
fig=plt.figure()
plt.title(particle+' Hillas Skewness')
plt.hist(t1s,label='Telescope 1',alpha=0.3)
plt.hist(t2s,label='Telescope 2',alpha=0.3)
plt.hist(t3s,label='Telescope 3',alpha=0.3)
plt.hist(t4s,label='Telescope 4',alpha=0.3)
plt.legend()
plt.xlabel('Value')
plt.ylabel('Frequency (counts)')
plt.savefig('../Figures/'+particle+'_hs.png')
fig=plt.figure()
plt.title(particle+' Hillas Kurtosis')
plt.hist(t1k,label='Telescope 1',alpha=0.3)
plt.hist(t2k,label='Telescope 2',alpha=0.3)
plt.hist(t3k,label='Telescope 3',alpha=0.3)
plt.hist(t4k,label='Telescope 4',alpha=0.3)
plt.legend()
plt.xlabel('Value')
plt.ylabel('Frequency (counts)')
plt.savefig('../Figures/'+particle+'_hk.png')
