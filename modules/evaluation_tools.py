


import numpy as np
from numba import jit

import math

from inference import make_particle_inference_dict

'''
outdict['t_mask'] =  truth[:,:,0:1]
        outdict['t_E']    =  truth[:,:,1:2]
        outdict['t_pos']  =  truth[:,:,2:4]
        outdict['t_ID']   =  truth[:,:,4:6]
        outdict['t_objidx']= truth[:,:,6:7]
        
        outdict['t_rhpos']= truth[:,:,7:10]
        outdict['t_rhid']= truth[:,:,10:11]
'''

tracker_pos_res = 22./4.

#matches reco to truth, per event
@jit(nopython=True)     
def c_find_best_matching_truth(pf_pos, pf_energy, pf_tidxs, t_Mask, t_pos, t_E, t_objidx,t_ID, rest_t_objidx, pos_tresh, en_thresh):
    # flags any used truth index with -1
    # returns per pf matched truth info. for non matched, returns -1
    
    
    #
    #
    # mask truth already used
    # save matching distance (without energy criterion)
    #
    # 
    #
    #
    # get unique truth only.
    # make sort of simultaneous matching, minimise distance (inkl energy) for all at once. 
    #
    # match all with pf_tidx>=0 first
    #
    # or use the ML truth. If track belongs to pf candidate -> unambiguous; mask out
    # then do cell matching with closest cell
    # 

    
    
    
    #first iteration for direct matching
    pf_ismatched = [False for _ in range(len(pf_pos))]
    matched_id = [0 for _ in range(len(pf_pos))]
    matched_e = [-1. for _ in range(len(pf_pos))]
    matched_posx = [0. for _ in range(len(pf_pos))]
    matched_posy = [0. for _ in range(len(pf_pos))]
    matched_tidxs = [-1 for _ in range(len(pf_pos))]
    
    t_ismatched=[]
    
    for i_pf in range(len(pf_pos)):
        pf_x = pf_pos[i_pf][0]
        pf_y = pf_pos[i_pf][1]
        pf_e = pf_energy[i_pf]
        pf_tidx = pf_tidxs[i_pf]
        

        #first see if a direct match can be found
        for i_t in range(len(t_pos)):
            if not t_Mask[i_t][0]: continue
            if pf_tidx == i_t: #match
                # fill all the stuff
                matched_id[i_pf]    = t_ID[i_t][0]
                matched_e[i_pf]     = t_E[i_t][0]
                matched_posx[i_pf]   = t_pos[i_t][0]
                matched_posy[i_pf]   = t_pos[i_t][1]
                matched_tidxs[i_pf] = t_objidx[i_t][0]
                
                t_ismatched.append(int(t_objidx[i_t][0]+0.1))
                #also flag this truth index as matched
                pf_ismatched[i_pf]=True
                break
    
    #determine remaining truth
    
    s_t_xs = []
    s_t_ys = []
    s_t_es = []
    s_t_ids = []
    s_t_tidx = []
    
    for i_t in range(len(t_pos)):
        t_mask = t_Mask[i_t][0]
        if t_mask < 1:
            continue
        t_idx = int(t_objidx[i_t][0]+0.1)
        if t_idx in s_t_tidx: continue
        if t_idx in t_ismatched: continue
        s_t_xs.append(t_pos[i_t][0])
        s_t_ys.append(t_pos[i_t][1])
        s_t_es.append(t_E[i_t][0])
        s_t_ids.append(t_ID[i_t][0])
        s_t_tidx.append(t_idx)
            
    # match the remaining truth -> reco
    
    for i_t in range(len(s_t_xs)):
        t_idx = s_t_tidx[i_t]
        if t_idx in t_ismatched: continue
        t_x = s_t_xs[i_t]
        t_y = s_t_ys[i_t]
        t_e = s_t_es[i_t]
        t_id = s_t_ids[i_t]
        
        bestmatch_distance_sq = 2*pos_tresh**2+1e6
        bestmatch_idx=-1
        
        for i_pf in range(len(pf_pos)):
            if pf_ismatched[i_pf]: continue
            pf_x = pf_pos[i_pf][0]
            pf_y = pf_pos[i_pf][1]
            pf_e = pf_energy[i_pf]
            
            dist_sq = (pf_x-t_x)**2 + (pf_y-t_y)**2
            if dist_sq > pos_tresh**2 : continue
            endiffsq = (pf_e/t_e -1)**2
            if endiffsq > en_thresh**2: continue
            dist_sq += (22./0.05)**2 * endiffsq
            if bestmatch_distance_sq < dist_sq : continue
            bestmatch_idx=i_pf
            
        if bestmatch_idx>=0:
            pf_ismatched[bestmatch_idx] = True
            matched_e[bestmatch_idx]     = s_t_es[i_t]
            matched_posx[bestmatch_idx]   = s_t_xs[i_t]
            matched_posy[bestmatch_idx]   = s_t_ys[i_t] 
            matched_tidxs[bestmatch_idx] = s_t_tidx[i_t] 
            t_ismatched.append(int(t_idx+0.1))
        
    
    #determine matching here  
    #for i_pf in range(len(pf_pos)):
    #    if pf_ismatched[i_pf]: continue #already matched
    #    pf_x = pf_pos[i_pf][0]
    #    pf_y = pf_pos[i_pf][1]
    #    pf_e = pf_energy[i_pf]
    #    
    #    bestmatch_distance_sq = 2*pos_tresh**2+1e6
    #    bestmatch_idx=-1
    #    
    #    for i_t in range(len(s_t_xs)):
    #        t_idx = s_t_tidx[i_t]
    #        if t_idx in t_ismatched: continue
    #        t_x = s_t_xs[i_t]
    #        t_y = s_t_ys[i_t]
    #        t_e = s_t_es[i_t]
    #        t_id = s_t_ids[i_t]
    #        
    #        dist_sq = (pf_x-t_x)**2 + (pf_y-t_y)**2
    #        if dist_sq > pos_tresh**2 : continue
    #        
    #        endiffsq = (pf_e/t_e -1)**2
    #        if endiffsq > en_thresh**2: continue
    #        dist_sq += (22./0.025)**2 * endiffsq
    #        if bestmatch_distance_sq < dist_sq : continue
    #        
    #        bestmatch_idx=i_t
    #        
    #    if bestmatch_idx>=0:
    #        pf_ismatched[i_pf] = True
    #        matched_id[i_pf]    = s_t_ids[bestmatch_idx] 
    #        matched_e[i_pf]     = s_t_es[bestmatch_idx]
    #        matched_posx[i_pf]   = s_t_xs[i_t]
    #        matched_posy[i_pf]   = s_t_ys[i_t] 
    #        matched_tidxs[i_pf] = s_t_tidx[bestmatch_idx] 
    #        t_ismatched.append(int(s_t_tidx[bestmatch_idx]+0.1))
            
    
    #leave only not matched 
    for i_iidx in range(len(rest_t_objidx)):
        if rest_t_objidx[i_iidx] in t_ismatched:
            rest_t_objidx[i_iidx] = -1
    
    
    not_recoed_pos=[]
    not_recoed_e=[] 
    not_recoed_id=[]
    #get truth for non matched
    for t_i in rest_t_objidx:
        if t_i < 0: continue
        for i_t in range(len(t_pos)):
            t_idx = t_objidx[i_t][0]
            t_mask = t_Mask[i_t][0]
            if t_mask < 1:
                continue #noise
            if not abs(t_idx - t_i)<0.1: continue
            t_x = t_pos[i_t][0]
            t_y = t_pos[i_t][1]
            t_e = t_E[i_t][0]
            t_id = t_ID[i_t][0]
            not_recoed_pos.append([t_x,t_y])
            not_recoed_e.append(t_e)
            not_recoed_id.append(t_id)
            break
            
            
        
            
    return matched_posx, matched_posy, matched_e, matched_id, not_recoed_pos, not_recoed_e, not_recoed_id
    
    
def multi_expand_dims(alist, axis):
    out=[]
    for a in alist:
        out.append(np.expand_dims(a,axis=axis))
    return out  


def make_evaluation_dict(array):
    #is_reco, reco_posx, reco_posy, reco_e, is_true, true_posx, true_posy, true_e, true_id
    out={}
    out['is_reco'] = array[:,0]
    out['reco_posx'] = array[:,1]
    out['reco_posy'] = array[:,2]
    out['reco_e'] = array[:,3]
    out['is_true'] = array[:,4]
    out['true_posx'] = array[:,5]
    out['true_posy'] = array[:,6]
    out['true_e'] = array[:,7]
    out['true_id'] = array[:,8]
    
    return out
    
    #is_reco, reco_posx, reco_posy, reco_e, is_true, true_posx, true_posy, true_e, true_id


#takes one event
def find_best_matching_truth_and_format(pf_pos, pf_energy, pf_tidx, truth): #two ecal cells
    
    '''
    returns p particle: is_reco, reco_posx, reco_posy, reco_e, is_true, true_posx, true_posy, true_e, true_id
    '''
    pos_tresh=3*22.
    en_thresh=0.9 #150% wrong energy
    d = make_particle_inference_dict(None, None, np.expand_dims(truth,axis=0))
    
    
    rest_t_objidx = np.unique(d['t_objidx'][0])
    n_true = float(len(rest_t_objidx)-1.)
    #pf_pos, pf_energy, pf_tidxs, t_Mask, t_pos, t_E, t_objidx,t_ID, rest_t_objidx, pos_tresh, en_thresh
    matched_posx, matched_posy, matched_e, matched_id, not_recoed_pos, not_recoed_e, not_recoed_id = c_find_best_matching_truth(
        pf_pos, pf_energy, pf_tidx, d['t_mask'][0],d['t_pos'][0], d['t_E'][0], d['t_objidx'][0], d['t_ID'][0],
                                                                     rest_t_objidx, pos_tresh, en_thresh)
    
    
    
    matched_posx,matched_posy    = np.array(matched_posx) , np.array(matched_posy)
    matched_e      = np.array(matched_e)
    matched_id     = np.array(matched_id)
    
    is_reco = np.zeros_like(matched_e)+1
    is_true = np.where(matched_e>=0, np.zeros_like(matched_e)+1., np.zeros_like(matched_e))
    
    n_true_arr = np.tile(n_true,[matched_e.shape[0]])
    #print('is_reco',is_reco.shape)
    #print('pf_pos',pf_pos.shape)
    #print('pf_energy',pf_energy.shape)
    #print('is_true',is_true.shape)
    #print('matched_posx',matched_posx.shape)
    #print('matched_posy',matched_posy.shape)
    #print('matched_e',matched_e.shape)
    #print('matched_id',matched_id.shape)
    
    all_recoed = multi_expand_dims([is_reco, pf_pos[:,0],pf_pos[:,1], pf_energy, 
                                 is_true, matched_posx,matched_posy, matched_e, matched_id, n_true_arr],axis=1)
    #concat the whole thing
    all_recoed = np.concatenate(all_recoed,axis=-1)
    
    
    if len(not_recoed_e):
        not_recoed_posx, not_recoed_posy = np.array(not_recoed_pos)[:,0], np.array(not_recoed_pos)[:,1]
        not_recoed_e   = np.array(not_recoed_e)
        not_recoed_id  = np.array(not_recoed_id)
        n_true_arr = np.tile(n_true,[not_recoed_e.shape[0]])
         
        all_not_recoed = multi_expand_dims([np.zeros_like(not_recoed_e)+1., not_recoed_posx, not_recoed_posy, not_recoed_e, not_recoed_id,n_true_arr], axis=1)
        #is_true, true_posx, true_posy, true_e, true_id
        all_not_recoed = np.concatenate(all_not_recoed,axis=-1)
        all_not_recoed = np.pad(all_not_recoed, [(0,0),(4,0)], mode='constant', constant_values=0)
        all_recoed = np.concatenate([all_recoed,all_not_recoed],axis=0)
    
    # particle: is_reco, reco_posx, reco_posy, reco_e, is_true, true_posx, true_posy, true_e, true_id, n_true
    return all_recoed
   

def determine_event_properties(event_particles, calo_energy=None):
    pufracs=[0.0, 0.2, 0.5, 0.8]
    names=""
    if event_particles is None:
        for pufrac in pufracs:
            pufracstr=str(pufrac)
            if len(names):
                names+=","
            names+="jet_mass_r_pu"+pufracstr+", jet_mass_t_pu"+pufracstr+", p_imbalance_x_r_pu"+pufracstr+", p_imbalance_x_t_pu"+pufracstr
        names+=',n_true, e_access_n'
        return names
    
    n_true = event_particles[0,9]
    n_part = len(event_particles)
    
    p_imbalance_x_r = np.sum(event_particles[:,0]*event_particles[:,1]*event_particles[:,3])
    p_imbalance_x_t = np.sum(event_particles[:,4]*event_particles[:,5]*event_particles[:,7])

    #simulate PU chs  using matched_id (8) and selecting a random fraction
    
    neutral_mask = event_particles[:,8] < 0.5 #everything not charged or without true match
    rarr = np.random.rand(n_part)
    
    
    out=[]
    
    for pufrac in pufracs:
        sel = rarr >= pufrac
        sel = np.logical_or(sel,neutral_mask) #always take neutrals
        
        jet_mass_r = np.sum(event_particles[:,0][sel]*event_particles[:,3][sel])
        jet_mass_t = np.sum(event_particles[:,4][sel]*event_particles[:,7][sel])
        
        p_imbalance_x_r = np.sum(event_particles[:,0][sel]*event_particles[:,1][sel]*event_particles[:,3][sel])
        p_imbalance_x_t = np.sum(event_particles[:,4][sel]*event_particles[:,5][sel]*event_particles[:,7][sel])
        
        out+=[jet_mass_r, jet_mass_t, p_imbalance_x_r, p_imbalance_x_t]
        pufracstr=str(pufrac)
        if len(names):
            names+=","
        names+="jet_mass_r_pu"+pufracstr+", jet_mass_t_pu"+pufracstr+", p_imbalance_x_r_pu"+pufracstr+", p_imbalance_x_t_pu"+pufracstr
    
    names+=',n_true, e_access_n'
    
    e_access_n = 0
    if calo_energy is not None:
        part_energy=np.sum(event_particles[:,0]*event_particles[:,3])
        #print(calo_energy, part_energy)
        e_access_n = max(0., calo_energy-part_energy)
    
    return np.array([out+[n_true, e_access_n]],dtype='float32'), names

def event_properties_dict(prop):
    out={}
    out['jet_mass_r']=prop[:,0]
    out['p_imbalance_x_r']=prop[:,1]
    out['p_imbalance_x_t']=prop[:,3]
    out['n_part']=prop[:,4]
    out['n_true']=prop[:,4]
    
    out['jet_mass_r']=prop[:,0]
    out['jet_mass_r']=prop[:,0]
    out['jet_mass_r']=prop[:,0]
    out['jet_mass_r']=prop[:,0]


def write_event_output_tree(allevents, names, outputFile):
    from root_numpy import array2root
    out = np.core.records.fromarrays(allevents.transpose() ,names=names)
    array2root(out, outputFile+"_events.root", 'tree')    
     
    
def write_output_tree(allparticles, outputFile):
    from root_numpy import array2root
    out = np.core.records.fromarrays(allparticles.transpose() ,names="is_reco, reco_posx, reco_posy, reco_e, is_true, true_posx, true_posy, true_e, true_id, n_true")
    array2root(out, outputFile+".root", 'tree')    
    
    
    
    