import ROOT
import numpy as np
from root_numpy import tree2array
from ROOT import gSystem, TFile, TTreeReader
from root_numpy import tree2array,root2array
import root_numpy
import matplotlib.pyplot as plt
from image_mapping import ImageMapper

ROOT.ROOT.EnableImplicitMT()

if gSystem.Load("$EVNDISPSYS/lib/libVAnaSum.so"):
    print("Problem loading EventDisplay libraries - please check this before proceeding")


inputfile='/lustre/fs19/group/cta/users/sspencer/ver/aux/64080_dst.root'
cam = 'VERITAS'

f=ROOT.TFile.Open(inputfile,'read')

mytree=f.Get("dst")

nochannels=499
notrigs=100

imarr=tree2array(mytree,branches=['sum'],start=0,stop=notrigs)
deadarr=tree2array(mytree,branches=['dead'],start=0,stop=notrigs)
evarr=tree2array(mytree,branches=['eventNumber'],start=0,stop=notrigs)

mappers = {}

hex_methods = ['oversampling', 'rebinning', 'nearest_interpolation',
               'bilinear_interpolation', 'bicubic_interpolation', 
               'image_shifting', 'axial_addressing']

hex_cams=['VERITAS']

for method in hex_methods:
    mapping_method = {cam: method for cam in hex_cams}
    mappers[method] = ImageMapper(mapping_method=mapping_method)

for i in np.arange(notrigs):
    fig,axs=plt.subplots(4,7,figsize=(30,24))
    axs=axs.ravel()
    j=0
    for method in hex_methods:
        image=imarr[i][0][0][:499].copy()
        image=np.expand_dims(image,1)
        image = mappers[method].map_image(image, cam)
        axs[j].pcolor(image[:,:,0], cmap='viridis')
        axs[j].set_title(method)
        axs[j].set_axis_off()
        deadvals=deadarr[i][0][0][:499]
        deadvals=np.expand_dims(deadvals,1)
        deadlist=np.where(deadvals!=0)[0]
        for k in deadlist:
            try:
                if imarr[i][0][0][k+1] !=0.0 and imarr[i][0][0][k-1] !=0.0:
                    imarr[i][0][0][k]=(imarr[i][0][0][k+1]+imarr[i][0][0][k-1])/2.0
                elif imarr[i][0][0][k+2] !=0.0 and imarr[i][0][0][k-2] !=0.0:
                    imarr[i][0][0][k]=(imarr[i][0][0][k+2]+imarr[i][0][0][k-2])/2.0
                elif imarr[i][0][0][k+3] !=0.0 and imarr[i][0][0][k-3] !=0.0:
                    imarr[i][0][0][k]=(imarr[i][0][0][k+3]+imarr[i][0][0][k-3])/2.0
                else:
                    print('Dead pixel not corrected: '+str(k))
                    continue
            except IndexError:
                print('Index error')
                continue
        im2=imarr[i][0][0][:499].copy()
        im2=np.expand_dims(im2,1)
        im2=mappers[method].map_image(im2,cam)
        axs[j+7].pcolor(im2[:,:,0],cmap='viridis')
        axs[j+7].set_axis_off()

        deadvals=mappers[method].map_image(deadvals,cam)
        axs[j+14].pcolor(deadvals[:,:,0],cmap='viridis')
        axs[j+14].set_axis_off()
        axs[j+21].pcolor((im2-image)[:,:,0],cmap='viridis')
        axs[j+21].set_axis_off()

        j=j+1
    plt.tight_layout()
    plt.show()

