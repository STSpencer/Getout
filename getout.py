import ROOT
import numpy as np

from ROOT import gSystem
if gSystem.Load("$EVNDISPSYS/lib/libVAnaSum.so"):
    print("Problem loading EventDisplay libraries - please check this before proceeding")


inputfile='/lustre/fs19/group/cta/users/sspencer/ver/user/64080.root'

f=ROOT.TFile.Open(inputfile,"read")
print(f)
