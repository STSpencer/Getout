import sys 
sys.argv.append( '-b-' )
import ROOT
import numpy as np
from root_numpy import tree2array
from ROOT import gSystem, TFile, TTreeReader
from root_numpy import tree2array,root2array,list_trees,list_branches
import root_numpy
import matplotlib.pyplot as plt
from image_mapping import ImageMapper
import h5py

ROOT.ROOT.EnableImplicitMT()
ROOT.gROOT.SetBatch(True)
if gSystem.Load("$EVNDISPSYS/lib/libVAnaSum.so"):
    print("Problem loading EventDisplay libraries - please check this before proceeding")

gammafile='/lustre/fs19/group/cta/users/sspencer/ver/aux/mcgammacrab67272.dst.root'
protonfile='/lustre/fs19/group/cta/users/sspencer/ver/aux/mcprotoncrab67272.dst.root' #This is a test tycho run just to use as a placeholder until we get some MC
cam = 'VERITAS'
runcode='Crab67272'
runname='/lustre/fs19/group/cta/users/sspencer/Crab67272/'+runcode+'_'
nochannels=499
nofiles=200

#print(list_branches(protonfile,'dst'))

f=ROOT.TFile.Open(protonfile,'read')
mytree=f.Get("dst")
evarr=tree2array(mytree,branches=['eventNumber'])
noprotons=len(evarr)
f.Close()

f=ROOT.TFile.Open(gammafile,'read')
mytree=f.Get("dst")
evarr=tree2array(mytree,branches=['eventNumber'])
nogammas=len(evarr)
f.Close()
print(noprotons,nogammas)
notrigs=min([noprotons,nogammas])
print('Number of triggers',notrigs)

mappers = {}
hex_methods = ['oversampling', 'rebinning', 'nearest_interpolation',
               'bilinear_interpolation', 'bicubic_interpolation', 
               'image_shifting','axial_addressing']

hex_cams=['VERITAS']


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
    if deadarr is not None:
        deadvals=deadarr[i][0][tel][:499]
        deadvals=np.expand_dims(deadvals,1)
        deadlist=np.where(deadvals!=0)[0]
        for k in deadlist:
            imarr=dedead(imarr,tel,k)
    try:
        #print(imarr[i][0][tel][:499])
        image=imarr[i][0][tel][:499]
    except IndexError:
        return None
    image=np.expand_dims(image,1)
    image = mappers[method].map_image(image, cam)
    return image


for method in hex_methods:
    mapping_method = {cam: method for cam in hex_cams}
    mappers[method] = ImageMapper(mapping_method=mapping_method)

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

        for method in hex_methods:
            image1=extract_im(mappers,i,method,None,imarr,0,cam) #Should set deadarr to None when MC comes through
            image2=extract_im(mappers,i,method,None,imarr,1,cam)
            image3=extract_im(mappers,i,method,None,imarr,2,cam)
            image4=extract_im(mappers,i,method,None,imarr,3,cam)
            if image1 is None and image2 is None and image3 is None and image4 is None:
                del to_hdf['id'][-1]
                del to_hdf['isGamma'][-1]
                break
            out_arr=np.zeros((4,np.shape(image1)[0],np.shape(image1)[1],1),dtype='float32')
            out_arr[0,:,:,:]=image1
            out_arr[1,:,:,:]=image2
            out_arr[2,:,:,:]=image3
            out_arr[3,:,:,:]=image4
            to_hdf[method].append(out_arr)

    f.Close()

    print('Processing Gammas')
    f=ROOT.TFile.Open(gammafile,'read')

    mytree=f.Get("dst")
    print('Gamma DST Retrieved')
    imarr=tree2array(mytree,branches=['sum'],start=startev,stop=stopev)
    evarr=tree2array(mytree,branches=['eventNumber'],start=startev,stop=stopev)
    mceng=tree2array(mytree,branches=['MCe0'],start=startev,stop=stopev)

    for i in np.arange(trigsperfile):
        
        to_hdf['id'].append(trigsperfile*(runno+1)+evarr[i][0])
        to_hdf['isGamma'].append(1)
        
        for method in hex_methods:
            image1=extract_im(mappers,i,method,None,imarr,0,cam)
            image2=extract_im(mappers,i,method,None,imarr,1,cam)
            image3=extract_im(mappers,i,method,None,imarr,2,cam)
            image4=extract_im(mappers,i,method,None,imarr,3,cam)
            if image1 is None and image2 is None and image3 is None and image4 is None:
                del to_hdf['id'][-1]
                del to_hdf['isGamma'][-1]
                break
            out_arr=np.zeros((4,np.shape(image1)[0],np.shape(image1)[1],1),dtype='float32')
            out_arr[0,:,:,:]=image1
            out_arr[1,:,:,:]=image2
            out_arr[2,:,:,:]=image3
            out_arr[3,:,:,:]=image4
            to_hdf[method].append(out_arr)

    f.Close()

    randomize = np.arange(len(to_hdf['isGamma'])) #Randomization Code
    np.random.shuffle(randomize)
    print('Writing HDF5')
    h5file = h5py.File(runname+str(runno)+'_SIM.hdf5',"w")
    
    for keyval in to_hdf.keys():
        if keyval=='id' or keyval=='isGamma':
            to_hdf[keyval]=np.asarray(to_hdf[keyval],dtype='int32')
        else:
            to_hdf[keyval]=np.asarray(to_hdf[keyval],dtype='float32')
        to_hdf[keyval]=to_hdf[keyval][randomize]
        h5file.create_dataset(keyval,data=to_hdf[keyval])

    h5file.close()
