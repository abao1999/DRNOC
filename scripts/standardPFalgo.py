#!/usr/bin/env python


from __future__ import print_function
import numpy as np
from argparse import ArgumentParser

from evaluation_tools import find_best_matching_truth_and_format, write_output_tree, determine_event_properties, write_event_output_tree
from PFTools import calo_getSeeds, calo_determineClusters, perform_linking, create_pf_tracks


parser = ArgumentParser('Performs standard PF reco and puts output into root file')
parser.add_argument('inputFile')
parser.add_argument('outputFile')

args=parser.parse_args()

from DeepJetCore.TrainData import TrainData


allparticles=[]
evt_prop=[]

nparticles=0
names=""

with open(args.inputFile) as file:
    for inputfile in file:
        inputfile = inputfile.replace('\n', '')
        if len(inputfile)<1: continue
        
        print('inputfile',inputfile)
        
        td = TrainData()
        td.readFromFile(inputfile)
        feat = td.transferFeatureListToNumpy()
        truth = td.transferTruthListToNumpy()[0]
        del td
        calo = feat[0]
        tracks = feat[1]
        
        print('making seeds, also using tracks as seeds')
        seed_idxs, seedmap = calo_getSeeds(calo,tracks)
        
        def run_event(event):
            
            #make calo clusters
            e_seeds = seed_idxs[event]
            e_calo = calo[event,:,:,:]
            e_tracks = tracks[event,:,:,:]
            
            PF_calo_clusters = calo_determineClusters(e_calo, e_seeds)
            PF_tracks = create_pf_tracks(e_tracks)
            
            pfcands = perform_linking(PF_calo_clusters,PF_tracks)
        
            ev_reco_E    = np.array([p.energy for p in pfcands])
            ev_reco_pos  = np.array([p.position for p in pfcands])
            pf_tidx = np.array([p.track_idx for p in pfcands])
            ev_truth = truth[event]
            
            
            arr=find_best_matching_truth_and_format(ev_reco_pos, ev_reco_E, pf_tidx, ev_truth)
            
            ev_pro, names = determine_event_properties(arr)
            #print(arr.shape)
            return (arr, ev_pro)
         
        out = []   
        useMP=True
        
        if useMP:
            from multiprocessing import Pool
            p = Pool(8)
            out = p.map(run_event,range(calo.shape[0])) 
            p.close()
        else:
            for event in range(len(calo)):
               out.append(run_event(event))
        #break    
        
        
 
        allparticles+=[p[0] for p in out]
        evt_prop+=[p[1] for p in out]
        #break
            
    
allparticles = np.concatenate(allparticles,axis=0)
evt_prop = np.concatenate(evt_prop,axis=0)
print('evt_prop',evt_prop.shape)
print(allparticles.shape)
print('efficiency: ', float(np.count_nonzero( allparticles[:,0] *  allparticles[:,4]))/float( np.count_nonzero(allparticles[:,4] ) ))
print('fake: ', float(np.count_nonzero( allparticles[:,0] *  (1.-allparticles[:,4])))/float( np.count_nonzero(allparticles[:,0] ) ))


write_output_tree(allparticles, args.outputFile)
names = determine_event_properties(None)
write_event_output_tree(evt_prop, names, args.outputFile)


