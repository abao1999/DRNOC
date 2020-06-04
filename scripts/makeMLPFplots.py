#!/usr/bin/env python

import numpy as np
from argparse import ArgumentParser
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt


parser = ArgumentParser('make plots')
parser.add_argument('inputFile')

args = parser.parse_args()

from DeepJetCore.TrainData import TrainData
from tools import make_particle_resolution_plots


#just read the stored data
td = TrainData()
td.readFromFile(args.inputFile)
indata = td.transferFeatureListToNumpy()
pred, feat, truth = indata[0],indata[1],indata[2]
del td

print('pred',pred.shape)
print('feat',feat.shape)
print('truth',truth.shape)

make_particle_resolution_plots(feat,pred,truth,outfile="reso.pdf")

