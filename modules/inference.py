
import math  
import numpy as np
from numba import jit

def make_inference_dict(pred,feat,truth):
    
    outdict = {}
    outdict['t_mask'] =  truth[:,:,:,0:1]
    outdict['t_pos']  =  truth[:,:,:,1:3]
    outdict['t_ID']   =  truth[:,:,:,3:6]
    outdict['t_dim']  =  truth[:,:,:,6:8]
    #n_objects = truth[:,0,0,8]

    outdict['p_beta']   =  pred[:,:,:,0:1]
    outdict['p_pos']    =  pred[:,:,:,1:3]
    outdict['p_ID']     =  pred[:,:,:,3:6]
    outdict['p_dim']    =  pred[:,:,:,6:8]
    
    outdict['p_ccoords'] = pred[:,:,:,8:]
    
    
    outdict['f_rgb'] = feat[:,:,:,0:3]
    outdict['f_xy'] = feat[:,:,:,3:]
    
    return outdict

'''

    p_beta    = Dense(1,activation='sigmoid')(x)
    p_tpos    = ScalarMultiply(10.)(Dense(2)(x))
    p_ID      = Dense(2,activation='softmax')(x)

    p_E       = ScalarMultiply(10.)(Dense(1)(x))
    p_ccoords = ScalarMultiply(10.)(Dense(2)(x))
    
    predictions=Concatenate()([p_beta , # 0 
                               p_E    ,  # 1
                               p_tpos   , #2,3
                               p_ID     , # 4,5
                               p_ccoords, # 6,7
                               ids,         #8 
                               energy_raw]) # 9, 10(posx), 11(posy)
'''

def make_particle_inference_dict(pred,feat,truth):
    
    outdict = {}
    
    if truth is not None:
        outdict['t_mask'] =  truth[:,:,0:1]
        outdict['t_E']    =  truth[:,:,1:2]
        outdict['t_pos']  =  truth[:,:,2:4]
        outdict['t_ID']   =  truth[:,:,4:6]
        outdict['t_objidx']= truth[:,:,6:7]
        
        outdict['t_rhpos']= truth[:,:,7:10]
        outdict['t_rhid']= truth[:,:,10:11]
    #n_objects = truth[:,0,0,8]

    if pred is not None:
        outdict['p_beta']      =  pred[:,:,0:1]
        outdict['p_E_corr']    =  pred[:,:,1:2]
        outdict['p_pos_offs']  =  pred[:,:,2:4]
        outdict['p_ID']        =  pred[:,:,4:6]
        
        outdict['p_ccoords'] = pred[:,:,6:8]
    
    if feat is not None:
        outdict['f_E'] = feat[:,:,0:1]
        outdict['f_pos'] = feat[:,:,1:3]
    
    
    return outdict
       
def maskbeta(datadict, dims, threshold):
    betamask = np.tile(datadict['p_beta'], [1,1,1,dims])
    betamask[betamask>threshold] = 1
    betamask[betamask<=threshold] = 0
    return betamask


@jit(nopython=True)        
def c_collectoverthresholds(betas, 
                            ccoords, 
                            sorting,
                            betasel,
                          beta_threshold, distance_threshold):
    

    for e in range(len(betasel)):
        selected = []
        for si in range(len(sorting[e])):
            i = sorting[e][si]
            use=True
            for s in selected:
                distance = math.sqrt( (s[0]-ccoords[e][i][0])**2 +  (s[1]-ccoords[e][i][1])**2 )
                if distance  < distance_threshold:
                    use=False
                    break
            if not use:
                betasel[e][i] = False
                continue
            else:
                selected.append(ccoords[e][i])
             
    return betasel
    
def collect_condensates(data, 
                          beta_threshold, distance_threshold):
    
    betas   = np.reshape(data['p_beta'], [data['p_beta'].shape[0], -1])
    ccoords = np.reshape(data['p_ccoords'], [data['p_ccoords'].shape[0], -1, data['p_ccoords'].shape[-1]])
    
    sorting = np.argsort(-betas, axis=1)
    
    betasel = betas > beta_threshold
    
    bsel =  c_collectoverthresholds(betas, 
                            ccoords, 
                            sorting,
                            betasel,
                          beta_threshold, distance_threshold)
    
    
    return np.reshape(bsel , data['p_beta'].shape)
    
