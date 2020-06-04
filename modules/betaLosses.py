

import tensorflow as tf
import keras
from keras import losses
import keras.backend as K

from tools import printWorkaround

#factorise a bit


#
#
#
#  all inputs/outputs of dimension B x V x F (with F being 1 in some cases)
#
#

def calulate_beta_scaling(d,minimum_confidence):
    return (1./(( 1. - d['p_beta'])+K.epsilon()) - 1.)**2 + minimum_confidence


def calulate_payload_beta_scaling(d, onset=0.5):
    beta_m_onset = tf.where(d['p_beta']>onset,d['p_beta']-onset,tf.zeros_like(d['p_beta']))
    beta_squeezed = (beta_m_onset)/(1-onset)
    return (1./(( 1. - beta_squeezed)+K.epsilon()) - 1.)**2

def create_pixel_loss_dict(truth, pred):
    '''
    input features as
    B x P x P x F
    with F = colours
    
    truth as 
    B x P x P x T
    with T = [mask, true_posx, true_posy, ID_0, ID_1, ID_2, true_width, true_height, n_objects]
    
    all outputs in B x V x 1/F form except
    n_active: B x 1
    
    '''
    outdict={}
    #truth = tf.Print(truth,[tf.shape(truth),tf.shape(pred)],'truth, pred ',summarize=30)
    def resh(lastdim):
        return (tf.shape(pred)[0],tf.shape(pred)[1]*tf.shape(pred)[2],lastdim)
    #make it all lists
    outdict['t_mask'] =  tf.reshape(truth[:,:,:,0:1], resh(1)) 
    outdict['t_pos']  =  tf.reshape(truth[:,:,:,1:3], resh(2), name="lala")
    outdict['t_ID']   =  tf.reshape(truth[:,:,:,3:6], resh(3))  
    outdict['t_dim']  =  tf.reshape(truth[:,:,:,6:8], resh(2))
    outdict['t_objidx']= tf.reshape(truth[:,:,:,8:9], resh(1))

    print('pred',pred.shape)

    outdict['p_beta']   =  tf.reshape(pred[:,:,:,0:1], resh(1))
    outdict['p_pos']    =  tf.reshape(pred[:,:,:,1:3], resh(2), name="lulu")
    outdict['p_ID']     =  tf.reshape(pred[:,:,:,3:6], resh(3))
    outdict['p_dim']    =  tf.reshape(pred[:,:,:,6:8], resh(2))
    
    outdict['p_ccoords'] = tf.reshape(pred[:,:,:,8:10], resh(2))
    
    flattened = tf.reshape(outdict['t_mask'],(tf.shape(outdict['t_mask'])[0],-1))
    outdict['n_nonoise'] = tf.expand_dims(tf.cast(tf.math.count_nonzero(flattened, axis=-1), dtype='float32'), axis=1)
    #will have a meaning for non toy model
    outdict['n_active'] = tf.zeros_like(outdict['n_nonoise'])+64.*64.
    outdict['n_noise'] = 64.*64.-outdict['n_nonoise']
    
    
    return outdict

def create_particle_loss_dict(truth, pred):
    '''
    input features as
    B x V x F  already reshaped in network
    with F = colours
    
    truth as 
    B x P x T
    with T = [mask, true_posx, true_posy, ID_0, ID_1, ID_2, true_width, true_height, n_objects]
    
    all outputs in B x V x 1/F form except
    n_active: B x 1
    
    '''
    outdict={}
    #truth = tf.Print(truth,[tf.shape(truth),tf.shape(pred)],'truth, pred ',summarize=30)

    #make it all lists
    outdict['t_mask'] =  truth[:,:,0:1]
    outdict['t_E']    =  truth[:,:,1:2]
    outdict['t_pos']  =  truth[:,:,2:4]
    outdict['t_ID']   =  truth[:,:,4:6]
    outdict['t_objidx']= truth[:,:,6:7]
    
    outdict['t_rhpos']= truth[:,:,7:10]
    outdict['t_rhid']= truth[:,:,10:11]

    print('pred',pred.shape)

    outdict['p_beta']   =  pred[:,:,0:1]
    outdict['p_E']      =  pred[:,:,1:2]
    outdict['p_pos']    =  pred[:,:,2:4]
    outdict['p_ID']     =  pred[:,:,4:6]
    
    outdict['p_ccoords'] = pred[:,:,6:8]
    
    outdict['p_rhid'] = pred[:,:,8:9]
    
    outdict['f_energy'] = pred[:,:,9:10]
    outdict['f_pos'] = pred[:,:,10:12]
    
    flattened = tf.reshape(outdict['t_mask'],(tf.shape(outdict['t_mask'])[0],-1))
    outdict['n_nonoise'] = tf.expand_dims(tf.cast(tf.math.count_nonzero(flattened, axis=-1), dtype='float32'), axis=1)
    #will have a meaning for non toy model
    #outdict['n_active'] = tf.zeros_like(outdict['n_nonoise'])+64.*64.
    outdict['n_noise'] =  tf.cast(tf.shape(outdict['t_mask'])[1], dtype='float32') -outdict['n_nonoise']
    outdict['n_total'] = outdict['n_noise']+outdict['n_nonoise']
    
    return outdict





def mean_nvert_with_nactive(A, n_active):
    '''
    n_active: B x 1
    A : B x V x F
    
    out: B x F
    '''
    assert len(A.shape) == 3
    assert len(n_active.shape) == 2
    den = n_active + K.epsilon()
    ssum = tf.reduce_sum(A, axis=1)
    return ssum / den

def beta_weighted_truth_mean(l_in, d, beta_scaling,Nobj):#l_in B x V x 1
    l_in = tf.reduce_sum(beta_scaling*d['t_mask']*l_in, axis=1)# B x 1
    print('Nobj',Nobj.shape)
    den =  tf.reduce_sum(d['t_mask']*beta_scaling, axis=1) + K.epsilon()#B x 1
    return l_in/den
    

def cross_entr_loss(d, beta_scaling,Nobj):
    tID = d['t_mask']*d['t_ID']
    tID = tf.where(tID<=0.,tf.zeros_like(tID)+10*K.epsilon(),tID)
    tID = tf.where(tID>=1.,tf.zeros_like(tID)+1.-10*K.epsilon(),tID)
    pID = d['t_mask']*d['p_ID']
    pID = tf.where(pID<=0.,tf.zeros_like(pID)+10*K.epsilon(),pID)
    pID = tf.where(pID>=1.,tf.zeros_like(pID)+1.-10*K.epsilon(),pID)
    
    xentr = d['t_mask']*beta_scaling * (-1.)* tf.reduce_sum(tID * tf.log(pID) ,axis=-1, keepdims=True)
    
    #xentr_loss = mean_nvert_with_nactive(d['t_mask']*xentr, d['n_nonoise'])
    #xentr_loss = tf.reduce_mean(tf.reduce_sum(d['t_mask']*xentr, axis = 1), axis=-1)
    return beta_weighted_truth_mean(xentr,d,beta_scaling,Nobj)
    return tf.reduce_mean(xentr_loss)

def pos_loss(d, beta_scaling,Nobj):
    posl = d['t_mask']*tf.reduce_sum(tf.abs(d['t_pos'] - d['p_pos']), axis=2,keepdims=True)
    #posl = tf.reduce_sum(posl, axis=1)
    #posl = mean_nvert_with_nactive(d['t_mask']*posl,d['n_nonoise'])
    #posl = tf.where(tf.is_nan(posl), tf.zeros_like(posl)+10., posl)
    
    
    return beta_weighted_truth_mean(posl,d,beta_scaling,Nobj)
    #return tf.reduce_mean( posl)


def k_L1_loss(d, beta_scaling, Nobj, kalpha, isobj, tdict, pdict):
    ppos = tf.gather_nd(d[pdict],kalpha,batch_dims=1)
    tpos = tf.gather_nd(d[tdict],kalpha,batch_dims=1) #B x 2
    balpha = tf.gather_nd(beta_scaling,kalpha,batch_dims=1) # B x 1
    
    posl = tf.reduce_sum(balpha*tf.abs(ppos - tpos), axis=-1)
    den = Nobj+K.epsilon()
    return tf.squeeze(isobj, axis=1)*posl/den

def k_pos_loss(d, beta_scaling, Nobj, kalpha, isobj):
    return k_L1_loss(d, beta_scaling, Nobj, kalpha, isobj, 't_pos', 'p_pos')

def k_dim_loss(d, beta_scaling, Nobj, kalpha, isobj):
    return k_L1_loss(d, beta_scaling, Nobj, kalpha, isobj, 't_dim', 'p_dim')


def k_xentr_loss(d, beta_scaling, Nobj, kalpha, isobj):
    pID = tf.gather_nd(d['p_ID'],kalpha,batch_dims=1)
    tID = tf.gather_nd(d['t_ID'],kalpha,batch_dims=1) #B x 3
    balpha = tf.gather_nd(beta_scaling,kalpha,batch_dims=1) # B x 1
    
    tID = tf.where(tID<=0.,tf.zeros_like(tID)+10*K.epsilon(),tID)
    tID = tf.where(tID>=1.,tf.zeros_like(tID)+1.-10*K.epsilon(),tID)

    pID = tf.where(pID<=0.,tf.zeros_like(pID)+10*K.epsilon(),pID)
    pID = tf.where(pID>=1.,tf.zeros_like(pID)+1.-10*K.epsilon(),pID)
    
    xentr = (-1.)* tf.reduce_sum(balpha * tID * tf.log(pID) ,axis=-1) #B
    
    den = Nobj+K.epsilon()
    return tf.squeeze(isobj, axis=1)*xentr/den


def k_energy_corr_loss(d, beta_scaling, Nobj, kalpha, isobj):
    pen = tf.gather_nd(d['p_E'],kalpha,batch_dims=1)
    ten = tf.gather_nd(d['t_E'],kalpha,batch_dims=1) #B x 1
    balpha = tf.gather_nd(beta_scaling,kalpha,batch_dims=1) # B x 1
    
    dE = pen-ten
    
    rel_calo_reso_sq =  ( (0.0028/tf.sqrt(ten+K.epsilon()))**2 + (0.12/(ten+K.epsilon()))**2 + 0.003**2 )
    pt = ten
    rel_trackreso = ( (pt/100.)*(pt/100.)*0.04 +0.01)
    comb_reso_sq =  ten**2 *0.5 *(rel_calo_reso_sq + rel_trackreso**2)#>0
    
    eloss = beta_scaling* dE**2/(comb_reso_sq+K.epsilon()) #B x 1
    eloss =tf.squeeze(eloss,axis=1)
    den = Nobj+K.epsilon()
    return tf.squeeze(isobj, axis=1)*eloss/den

def box_loss(d, beta_scaling,Nobj):
    bboxl = tf.reduce_sum(tf.abs(d['t_dim'] - d['p_dim']), axis=-1,keepdims=True)
    #bboxl = tf.reduce_sum(bboxl, axis=1)
    #bboxl = mean_nvert_with_nactive(d['t_mask']*bboxl,d['n_nonoise'])
    #bboxl = tf.where(tf.is_nan(bboxl), tf.zeros_like(bboxl)+10., bboxl)
    #return bboxl
    return beta_weighted_truth_mean(bboxl,d,beta_scaling,Nobj)
    
    #return tf.reduce_mean( bboxl)
    
def part_pos_loss(d, beta_scaling,Nobj):
    f_pos = d['f_pos']
    dPos = d['p_pos']+f_pos - d['t_pos'] #B x V x 2
    posl = tf.reduce_sum( dPos**2, axis=-1, keepdims=True )#B x V x 1
    
    return beta_weighted_truth_mean(posl,d,beta_scaling,Nobj)
    #return tf.reduce_mean( posl)


def energy_corr_loss(d,payload_scaling,Nobj):
    f_en = d['f_energy']
    dE = d['p_E']*f_en- d['t_E']
    
    #rel_calo_reso_sq =  d['t_mask']*( (0.0028/tf.sqrt(d['t_E']+K.epsilon()))**2 + (0.12/(d['t_E']+K.epsilon()))**2 + 0.003**2 )
    #pt = d['t_E']
    #rel_trackreso = d['t_mask']*( (pt/100.)*(pt/100.)*0.04 +0.01)
    #comb_reso_sq =  d['t_E']**2 *0.5 *(rel_calo_reso_sq + rel_trackreso**2)#>0
    #
    #eloss = d['t_mask']*payload_scaling* dE**2/(comb_reso_sq+K.epsilon())
    
    eloss = dE**2/(d['t_E']**2 + K.epsilon())
    
    return beta_weighted_truth_mean(eloss,d,payload_scaling,Nobj)#>0
    
        
def sup_noise_loss(d):
    return tf.reduce_mean(mean_nvert_with_nactive(((1.-d['t_mask'])*d['p_beta']), 
                                            tf.abs(d['n_active']-d['n_nonoise']))  )  


def calculate_charge(beta, q_min):
    beta = tf.clip_by_value(beta,0,1-K.epsilon()) #don't let gradient go to nan
    return tf.atanh(beta)+q_min


def sub_object_condensation_loss(d,q_min,Ntotal=4096):
    
    q = calculate_charge(d['p_beta'],q_min)
    
    L_att = tf.zeros_like(q[:,0,0])
    L_rep = tf.zeros_like(q[:,0,0])
    L_beta = tf.zeros_like(q[:,0,0])
    
    Nobj = tf.zeros_like(q[:,0,0])
    
    isobj=[]
    alpha=[]
    
    for k in range(9):#maximum number of objects
        
        Mki      = tf.where(tf.abs(d['t_objidx']-float(k))<0.2, tf.zeros_like(q)+1, tf.zeros_like(q))
        
        print('Mki',Mki.shape)
        
        iobj_k   = tf.reduce_max(Mki, axis=1) # B x 1
        
        
        Nobj += tf.squeeze(iobj_k,axis=1)
        
        
        kalpha   = tf.argmax(Mki*d['t_mask']*d['p_beta'], axis=1)
        
        isobj.append(iobj_k)
        alpha.append(kalpha)
        
        print('kalpha',kalpha.shape)
        
        x_kalpha = tf.gather_nd(d['p_ccoords'],kalpha,batch_dims=1)
        x_kalpha = tf.expand_dims(x_kalpha, axis=1)
        print('x_kalpha',x_kalpha.shape)
        
        q_kalpha = tf.gather_nd(q,kalpha,batch_dims=1)
        q_kalpha = tf.expand_dims(q_kalpha, axis=1)
        
        distance  = tf.sqrt(tf.reduce_sum( (x_kalpha-d['p_ccoords'])**2, axis=-1 , keepdims=True)+K.epsilon()) #B x V x 1
        F_att     = q_kalpha * q * distance**2 * Mki
        F_rep     = q_kalpha * q * tf.nn.relu(1. - distance) * (1. - Mki)
        
        L_att  += tf.squeeze(iobj_k * tf.reduce_sum(F_att, axis=1), axis=1)/(Ntotal)
        L_rep  += tf.squeeze(iobj_k * tf.reduce_sum(F_rep, axis=1), axis=1)/(Ntotal)
        
        
        beta_kalpha = tf.gather_nd(d['p_beta'],kalpha,batch_dims=1)
        L_beta += tf.squeeze(iobj_k * (1-beta_kalpha),axis=1)
        
        
    L_beta/=Nobj
    #L_att/=Nobj
    #L_rep/=Nobj
    
    L_suppnoise = tf.squeeze(tf.reduce_sum( (1.-d['t_mask'])*d['p_beta'] , axis=1) / (d['n_noise'] + K.epsilon()), axis=1)
    
    reploss = tf.reduce_mean(L_rep)
    attloss = tf.reduce_mean(L_att)
    betaloss = tf.reduce_mean(L_beta)
    supress_noise_loss = tf.reduce_mean(L_suppnoise)
    
    return reploss, attloss, betaloss, supress_noise_loss, Nobj, isobj, alpha
    

def object_condensation_loss(truth,pred):
    d = create_pixel_loss_dict(truth,pred)
    
    reploss, attloss, betaloss, supress_noise_loss, Nobj, isobj, alpha = sub_object_condensation_loss(d,q_min=1.)

    
    payload_scaling = calculate_charge(d['p_beta'],0.1)
    
    posl =   tf.zeros_like(isobj[0][:,0])#B 
    bboxl = posl
    xentr_loss = posl
    
    
    for i in range(0):
        kalpha = alpha[i]
        iobj_k = isobj[i]
        
        posl += k_pos_loss(d, payload_scaling, Nobj, kalpha, iobj_k) 
        bboxl += k_dim_loss(d, payload_scaling, Nobj, kalpha, iobj_k)
        xentr_loss += k_xentr_loss(d, payload_scaling, Nobj, kalpha, iobj_k)
    
    #maybe these should be WEIGHTED mean with scaling, more control
    posl       =  0.*tf.reduce_mean(pos_loss(d,payload_scaling,Nobj))/ 16.
    bboxl      = 0.*tf.reduce_mean(box_loss(d,payload_scaling,Nobj)) / 8.
    xentr_loss =  tf.reduce_mean(cross_entr_loss(d,payload_scaling,Nobj))
    
    
    
    loss = reploss + attloss + betaloss + supress_noise_loss + posl + bboxl + xentr_loss
    
    loss = tf.Print(loss,[loss,
                              reploss,
                              attloss,
                              betaloss,
                              supress_noise_loss,
                              posl,
                              bboxl,
                              xentr_loss
                              ],
                              'loss, repulsion_loss, attraction_loss, min_beta_loss, supress_noise_loss, pos_loss, bboxl, xentr_loss  ' )
    return loss
    

def particle_condensation_loss(truth,pred):
    
    #truth = tf.Print(truth,[truth],'truth ',summarize=3*1280)
    
    d = create_particle_loss_dict(truth,pred)
    
    
    
    reploss, attloss, betaloss, supress_noise_loss, Nobj,isobj, alpha = sub_object_condensation_loss(d,q_min=.1,Ntotal=d['n_total'])
    
    
    ididff = tf.concat([d['p_rhid'], d['t_rhid'], d['p_rhid']-d['t_rhid'] ], axis=-1)
    
    #reploss = tf.Print(reploss,[tf.reduce_mean(d['p_rhid']-d['t_rhid'])],'ididff ')
    
    
    payload_scaling = calculate_charge(d['p_beta'],0.)
    
    posl =   tf.zeros_like(isobj[0][:,0])#B 
    E_loss = posl
    xentr_loss = posl
    
    
    for i in range(0):
        kalpha = alpha[i]
        iobj_k = isobj[i]
        
        #posl += k_pos_loss(d, payload_scaling, Nobj, kalpha, iobj_k) 
        #E_loss += k_energy_corr_loss(d, payload_scaling, Nobj, kalpha, iobj_k)
        #xentr_loss += k_xentr_loss(d, payload_scaling, Nobj, kalpha, iobj_k)
        
    
    posl       = 0.01*  tf.reduce_mean(part_pos_loss(d,payload_scaling,Nobj))
    E_loss     = 20.* tf.reduce_mean(energy_corr_loss(d,payload_scaling,Nobj))
    #xentr_loss = 0.0* tf.reduce_mean(cross_entr_loss(d,payload_scaling,Nobj))
    
    #betaloss *= 10.
    
    loss = reploss + attloss + betaloss + supress_noise_loss + posl + E_loss
    
    loss = printWorkaround(loss, [d['n_noise']],'n_noise ')
    
    loss = printWorkaround(loss,[loss,
                              reploss,
                              attloss,
                              betaloss,
                              supress_noise_loss,
                              E_loss,
                              tf.sqrt(E_loss/10),
                              posl,
                              tf.sqrt(posl)*10
                              ],
                              'loss, repulsion_loss, attraction_loss, min_beta_loss, supress_noise_loss, E_loss, sqrt(E_loss/10), posl , tf.sqrt(posl)*10[cm]' )
    return loss

