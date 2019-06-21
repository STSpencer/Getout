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
#f=TFile.Open(inputfile,"read")

def Map(tf, browsable_to, tpath=None):
    """
    Maps objets as dict[obj_name][0] using a TFile (tf) and TObject to browse.
    """
    m = {}
    for k in browsable_to.GetListOfKeys():
        n = k.GetName()
        if tpath == None:
            m[n] = [tf.Get(n)]
        else:
            m[n] = [tf.Get(tpath + "/" + n)]
    return m

def Expand_deep_TDirs(tf, to_map, tpath=None):
    """
    A recursive deep-mapping function that expands into TDirectory(ies)
    """
    names = sorted(to_map.keys())
    for n in names:
        if len(to_map[n]) != 1:
            continue
        if tpath == None:
            tpath_ = n
        else:
            tpath_ = tpath + "/" + n
        
        tobject = to_map[n][0]
        if type(tobject) is ROOT.TDirectoryFile:
            m = Map(tf, tobject, tpath_)
            to_map[n].append(m)
            Expand_deep_TDirs(tf, m, tpath_)

def Map_TFile(filename, deep_maps=None):
    """
    Maps an input file as TFile into a dictionary(ies) of objects and names.
    Structure: dict[name][0] == object, dict[name][1] == deeper dict.
    """
    if deep_maps == None:
        deep_maps = {}
    if not type(deep_maps) is dict:
        return deep_maps
    
    f = ROOT.TFile(filename)
    m = Map(f, f)
    Expand_deep_TDirs(f, m)

    deep_maps[filename] = [f]
    deep_maps[filename].append(m)
    
    return deep_maps


def plot_image(image):
    fig, ax = plt.subplots(1)
    ax.set_aspect(1)
    ax.pcolor(image[:,:,0], cmap='viridis')
    plt.show()

#print(Map_TFile(inputfile))
print(root_numpy.list_directories(inputfile))
print(root_numpy.list_trees(inputfile))
blist=root_numpy.list_branches(inputfile,'dst')
print(blist)
#hist=root_numpy.evaluate(inputfile,'pulseSums/hSumHigh_1_0')
#arr=root_numpy.root2array(inputfile,'pulseSums')
#print(arr)
#raise KeyboardInterrupt
f=ROOT.TFile.Open(inputfile,'read')
#print(dir(f))
mytree=f.Get("dst")
#print(dir(mytree))
nochannels=499
notrigs=100
'''
notrigs=len(f.Get('pulseSums/hSumHigh_1_0'))
data_arr=np.zeros((notrigs,nochannels))

for i in np.arange(nochannels):
    j=0
    evns=f.Get('pulseSums/hSumHigh_1_'+str(i))
    for pixel in evns:
        data_arr[j][i]=pixel
        j=j+1 

print(data_arr,np.shape(data_arr))
#print(mytree.Print('hSumHigh_1_0'))


print(dir(mytree))
print(type(mytree))
print(mytree)
for x in blist:
    arr=tree2array(mytree,branches=[x],start=0,stop=1)
    print(x,np.shape(arr[0][0]))
    print(arr[0])
raise KeyboardInterrupt
imarr=tree2array(mytree,branches=['sum'],start=0,stop=10)
deadarr=tree2array(mytree,branches=['dead'],start=0,stop=10)
print(np.shape(imarr))
'''
imarr=tree2array(mytree,branches=['sum'],start=0,stop=notrigs)
print(imarr)
print(np.shape(imarr))
print(imarr[0][0])
print(np.shape(imarr[0][0][0]))

mappers = {}
print("Initialization time (total for all telescopes):")
hex_methods = ['oversampling', 'rebinning', 'nearest_interpolation',
               'bilinear_interpolation', 'bicubic_interpolation', 
               'image_shifting', 'axial_addressing']
hex_cams=['VERITAS']
for method in hex_methods:
    mapping_method = {cam: method for cam in hex_cams}
    mappers[method] = ImageMapper(mapping_method=mapping_method)

for i in np.arange(notrigs):
    fig,axs=plt.subplots(1,7,figsize=(30,4))
    axs=axs.ravel()
    j=0
    for method in hex_methods:
        print(imarr[i][0])
        image=imarr[i][0][0][:499]
        print('im',image)
        print(np.shape(image))
        image=np.expand_dims(image,1)
        print('{}: {}'.format(cam, method))
        print(np.shape(image))
        image = mappers[method].map_image(image, cam)
        axs[j].pcolor(image[:,:,0], cmap='viridis')
        axs[j].set_title(method)
        j=j+1
    plt.tight_layout()
    plt.show()
    #plt.imshow(image)
    #plt.show()

