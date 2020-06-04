#!/usr/bin/env python


from __future__ import print_function
import numpy as np
from DeepJetCore.TrainData import TrainData
from argparse import ArgumentParser

from evaluation_tools import find_best_matching_truth_and_format, write_output_tree, determine_event_properties, write_event_output_tree
from inference import make_particle_inference_dict,  collect_condensates

parser = ArgumentParser('Selects condensates and puts output into root file')
parser.add_argument('inputFile')
parser.add_argument('outputFile')


args=parser.parse_args()


allparticles=[]
all_ev_prop=[]
names=""

with open(args.inputFile) as file:
    for inputfile in file:
        inputfile = inputfile.replace('\n', '')
        if len(inputfile)<1: continue

        print('inputfile',inputfile)
        
        td = TrainData()
        td.readFromFile(inputfile)
        indata = td.transferFeatureListToNumpy()
        pred, feat, truth = indata[0],indata[1],indata[2]
        del td
        
        d = make_particle_inference_dict(pred, feat, truth)
        condensate_mask = np.squeeze(collect_condensates(d, 0.1, 0.8),axis=2) #B x V x 1
        
        pred_E   = d['f_E']* d['p_E_corr']
        pred_pos = d['f_pos'] + d['p_pos_offs']
        calo_energy = None #not supported by data formet..
        #np.sum(d['f_E'][:,0:16*16,0],axis=-1)#calo energy
        #loop over events here.. easier
        
        nevents = pred.shape[0]
        all_idxs = np.array([i for i in range(pred.shape[1])])
        
        #print('pred_pos',pred_pos.shape)
        #print('all_idxs',all_idxs.shape)
        
        for event in range(nevents):
            ev_pred_E    = pred_E[event][condensate_mask[event]>0][:,0]
            ev_pred_pos  = pred_pos[event][condensate_mask[event]>0]
            ev_truth = truth[event]
            ob_idx = all_idxs[condensate_mask[event]>0]
            
            eventparticles = find_best_matching_truth_and_format(ev_pred_pos, ev_pred_E, ob_idx, ev_truth)
            allparticles.append(eventparticles)
            
            ev_pro, names = determine_event_properties(eventparticles, None)
            all_ev_prop.append(ev_pro)
    
    

allparticles = np.concatenate(allparticles,axis=0)
all_ev_prop = np.concatenate(all_ev_prop,axis=0)
print(all_ev_prop.shape)

#is_reco, reco_posx, reco_posy, reco_e, is_true, true_posx, true_posy, true_e, true_id
print('efficiency: ', float(np.count_nonzero( allparticles[:,0] *  allparticles[:,4]))/float( np.count_nonzero(allparticles[:,4] ) ))
print('fake: ', float(np.count_nonzero( allparticles[:,0] *  (1.-allparticles[:,4])))/float( np.count_nonzero(allparticles[:,0] ) ))

write_output_tree(allparticles, args.outputFile)
write_event_output_tree(all_ev_prop, names, args.outputFile)






