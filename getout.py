import ROOT
import numpy as np
from root_numpy import tree2array
from ROOT import gSystem, TFile, TTreeReader
from root_numpy import tree2array
import matplotlib.pyplot as plt

ROOT.ROOT.EnableImplicitMT()

if gSystem.Load("$EVNDISPSYS/lib/libVAnaSum.so"):
    print("Problem loading EventDisplay libraries - please check this before proceeding")


inputfile='/lustre/fs19/group/cta/users/sspencer/ver/user/64080.root'

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

print(Map_TFile(inputfile))
f=ROOT.TFile(inputfile)
print(dir(f))
mytree=f.Get("Tel_1/dbpixeldata_Currents")
imarr=tree2array(mytree)
#print(imarr,type(imarr))
for i in np.arange(len(imarr)):
    image=imarr[i][3]
    image.reshape((25,25))
    print(np.shape(image))
    #plt.imshow(image)
    #plt.show()

