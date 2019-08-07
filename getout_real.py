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

datafile='/lustre/fs19/group/cta/users/sspencer/ver/eventdisplay_output/64080.dst.root'
cam = 'VERITAS'

f=ROOT.TFile.Open(datafile,'read')
mytree=f.Get('dst')
evarr=tree2array(mytree,branches=['eventNumber'])
f.Close()
notrigs=len(evarr)

print(notrigs)
runcode=64080
nochannels=499
nofiles=100
runname='/lustre/fs19/group/cta/users/sspencer/Real/'+str(runcode)+'_'
mappers = {}

hex_methods = ['oversampling', 'rebinning', 'nearest_interpolation',
               'bilinear_interpolation', 'bicubic_interpolation', 
               'image_shifting', 'axial_addressing']

hex_cams=['VERITAS']

to_hdf={'id':[],'oversampling':[],'rebinning':[],'nearest_interpolation':[],'bilinear_interpolation':[],'bicubic_interpolation':[],'image_shifting':[],'axial_addressing':[]}

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

def extract_im(mappers,i,method,deadarr,imarr,tel,cam):
    '''Extracts VERITAS images from root files and performs hexagonal image manipulation.'''
    deadvals=deadarr[i][0][tel][:499]
    deadvals=np.expand_dims(deadvals,1)
    deadlist=np.where(deadvals!=0)[0]
    for k in deadlist:
        imarr=dedead(imarr,tel,k)
    image=imarr[i][0][tel][:499]
    image=np.expand_dims(image,1)
    image = mappers[method].map_image(image, cam)
    return image


for method in hex_methods:
    mapping_method = {cam: method for cam in hex_cams}
    mappers[method] = ImageMapper(mapping_method=mapping_method)

trigsperfile=int(float(notrigs)/float(nofiles))

for runno in np.arange(nofiles)+1:
    to_hdf={'id':[],'oversampling':[],'rebinning':[],'nearest_interpolation':[],'bilinear_interpolation':[],'bicubic_interpolation':[],'image_shifting':[],'axial_addressing':[]}
    startev=runno*trigsperfile
    if runno==nofiles:
        stopev=notrigs
    else:
        stopev=(runno+1)*trigsperfile
    print('Processing Data: '+str(runno))
    f=ROOT.TFile.Open(datafile,'read')
    mytree=f.Get("dst")
    
    imarr=tree2array(mytree,branches=['sum'],start=startev,stop=stopev)
    deadarr=tree2array(mytree,branches=['dead'],start=startev,stop=stopev)
    evarr=tree2array(mytree,branches=['eventNumber'],start=startev,stop=stopev)
    print(startev,stopev,trigsperfile,runno,evarr)
    for i in np.arange(trigsperfile):
        to_hdf['id'].append(evarr[i][0])
        for method in hex_methods:
            image1=extract_im(mappers,i,method,deadarr,imarr,0,cam) #Should set deadarr to None when MC comes through
            image2=extract_im(mappers,i,method,deadarr,imarr,1,cam)
            image3=extract_im(mappers,i,method,deadarr,imarr,2,cam)
            image4=extract_im(mappers,i,method,deadarr,imarr,3,cam)
            out_arr=np.zeros((4,np.shape(image1)[0],np.shape(image1)[1],1),dtype='float32')
            out_arr[0,:,:,:]=image1
            out_arr[1,:,:,:]=image2
            out_arr[2,:,:,:]=image3
            out_arr[3,:,:,:]=image4
            to_hdf[method].append(out_arr)

    f.Close()

    print('Writing HDF5')
    h5file = tables.open_file(runname+str(runno)+'_REAL.hdf5', mode="w")
    root = h5file.root
    
    for keyval in to_hdf.keys():
        if keyval=='id':
            to_hdf[keyval]=np.asarray(to_hdf[keyval],dtype='int32')
        else:
            to_hdf[keyval]=np.asarray(to_hdf[keyval],dtype='float32')
        h5file.create_array(root,keyval,to_hdf[keyval],keyval)

    h5file.close()



