import ROOT
import numpy as np
from root_numpy import tree2array
from ROOT import gSystem, TFile, TTreeReader
from root_numpy import tree2array,root2array
import root_numpy
import matplotlib.pyplot as plt
from image_mapping import ImageMapper
import tables

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

    for method in hex_methods:
        image=imarr[i][0][0][:499]
        image=np.expand_dims(image,1)
        image = mappers[method].map_image(image, cam)
        

    for method in hex_methods:
        image=imarr[i][0][1][:499]
        image=np.expand_dims(image,1)
        image = mappers[method].map_image(image, cam)

    for method in hex_methods:
        image=imarr[i][0][2][:499]
        image=np.expand_dims(image,1)
        image = mappers[method].map_image(image, cam)

    for method in hex_methods:
        image=imarr[i][0][3][:499]
        image=np.expand_dims(image,1)
        image = mappers[method].map_image(image, cam)

