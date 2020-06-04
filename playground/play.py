import numpy as np
from DeepJetCore.TrainData import TrainData
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import math  
from numba import jit
from inference import collect_condensates   , make_inference_dict

td = TrainData()
#td.readFromFile("../results_partial/predictions/pred_9.djctd")
td.readFromFile("../data/test_data/9.djctd")
td.x = td.transferFeatureListToNumpy()
td.y = td.transferTruthListToNumpy()
td.z = td.transferWeightListToNumpy()

x = td.x
y = td.y
z = td.z

print(len(x))

print(x[0].shape)
print(x[1].shape)
print(x[2].shape)
#print(y.shape)
#print(z.shape)

data = make_inference_dict(td.x[0],td.x[1],td.x[2])

#outfile = infile[:-5]
#print(outfile)

#from DeepJetCore.TrainData import TrainData

#tdnew = traind()
#tdnew._store(x,y,w)
#tdnew.writeToFile(outfile+".djctd")
