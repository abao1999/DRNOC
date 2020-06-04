



from DeepJetCore.TrainData import TrainData, fileTimeOut
import numpy as np
import uproot
from numba import jit

@jit(nopython=True)           
def c_find_truth_index(toten, energies, sel_energies):
    for o in range(toten.shape[0]):#object
        for i in range(energies.shape[1]):#hits
            if energies[o,i]*100./5. < toten[o,0]: # at least 5% of the shower energy to be considered (used to be 1)
                sel_energies[o,i]=0
    return sel_energies

#@jit(nopython=True)   
def c_sum_energies(energies, uni_t_idx, out_energies):#energies: Nev x V, t_idx: Nev, out_energies=0, out_energies: V
    #maxtidx = np.max(uni_t_idx)
    for i in uni_t_idx:
        if i<0: continue
        for v in range(energies.shape[1]):
            out_energies[v] += energies[i,v]
            
    return out_energies
    
class TrainData_PF(TrainData):
    def __init__(self):
        TrainData.__init__(self)
        self.npart=9
    
    def separateLayers(self, inarr, layerarr):
        #lay0 = np.array(list(inarr[layerarr<-99.9]), dtype='float32')
        track = inarr[layerarr<0]#np.array(list(inarr[layerarr<0]), dtype='float32')
        calo = inarr[layerarr>=0]#np.array(list(inarr[layerarr>=0]), dtype='float32')
        return calo, track
        
    def merge_features(self, 
                       intuple_energy,
                       intuple_x,
                       intuple_y,
                       intuple_z):
        pass
    
    
    def tonumpy(self,inarr):
        return np.array(list(inarr), dtype='float32')
    
    
    
    def find_truth_index(self, energies): #energies is Nev x V
        #get total energy
        toten = np.sum(energies,axis=-1, keepdims=True)/2. #tracks have energy # Nev x 1
        #remove lower than 1% from indexing
        sel_energies=np.array(energies)
        sel_energies=c_find_truth_index(toten, energies, sel_energies)
        
        #don't use simple argmax
        i_cce = np.transpose(sel_energies, axes=[1,0])
        t_idx = np.argmax(i_cce, axis=-1) # V
        t_idx    = np.where(np.sum(sel_energies,axis=0)>0.0001, t_idx, np.zeros_like(t_idx,dtype='float32')-1.)
        return t_idx
        #if one is not present at all, discard?
        #some indices might not be present
        
    #only sum those that have truth association!   
     
    def sum_energies(self, energies, uni_t_idx, out_energies):#energies: Nev x V, t_idx: Nev, out_energies=0, out_energies: V
        return c_sum_energies(energies, uni_t_idx, out_energies)
    
    #@jit(nopython=True) 
    def mergeShowers(self, 
                     isElectron,
                     isGamma,
                     isPositron,
                     true_energy,
                     true_x,
                     true_y,
                     
                     rechit_energy,
                     rechit_x,
                     rechit_y,
                     rechit_z,
                     rechit_layer,
                     rechit_detid,
                     
                     maxpart,
                     istraining
                     ):
        
        
        #cc_rechit_energy = np.expand_dims(rechit_energy, axis=2)
        
        i_rhx = np.expand_dims(rechit_x[0], axis=1) #same for every event
        i_rhy = np.expand_dims(rechit_y[0], axis=1)
        i_rhz = np.expand_dims(rechit_z[0], axis=1)
        i_rhl = np.expand_dims(rechit_layer[0], axis=1)
        i_rhid = np.expand_dims(rechit_detid[0], axis=1)
        
        totalevents = rechit_energy.shape[0]
        used_events = 0
        
        feat=[]
        truth=[]
        layer=[]
        npart_arr=[]
        truth_en=[]
        
        while used_events < totalevents:
            npart=1
            tot_true_e=0
            if maxpart>1:
                npart = np.random.randint(1,maxpart)
                if not istraining: #flat distribution
                    probs = [1/float(i) for i in range(1,maxpart+1)]
                    #probs.reverse()
                    probs = np.array(probs)
                    probs/= np.sum(probs)
                    
                    npart = np.random.choice(np.array(range(1,maxpart+1)), p= probs)
            if maxpart<0: #jets
                npart = np.random.poisson(np.random.randint(1,-maxpart))+1
                
                
            if used_events+npart > totalevents:
                npart=totalevents-used_events
                
            i_cce = rechit_energy[used_events:used_events+npart] # npart x 400xx
            
            #only associate those with at least 5% of total energy of that shower in the cell
            
            t_idx = self.find_truth_index(i_cce)
            t_mask   = np.where(t_idx>=0, np.zeros_like(t_idx,dtype='float32')+1., np.zeros_like(t_idx,dtype='float32'))
            
            u_t_idx = np.array(np.unique(t_idx),dtype='int64').tolist()
            #print(u_t_idx)
            
            # only consider on truth those that actually have at least one shower associated as truth 
            # so that fully overlapping showers are removed -> ideal reconstruction possible!
            
            esum = np.zeros(i_cce.shape[1],dtype='float32')
            esum = self.sum_energies(i_cce, u_t_idx, esum)
        
            #print(esum)
        
            i_feat = np.concatenate([np.expand_dims(esum, axis=1),
                                   i_rhx, 
                                   i_rhy, 
                                   i_rhz, 
                                   i_rhl,
                                   i_rhid
                                   ],axis=-1)
            
            
            #get truth index etc
            
            t_energy = np.zeros_like(esum,dtype='float32')
            t_x = np.zeros_like(esum,dtype='float32')
            t_y = np.zeros_like(esum,dtype='float32')
            
            t_iselectron = np.zeros_like(esum,dtype='float32')
            t_isgamma = np.zeros_like(esum,dtype='float32')
            t_ispositron = np.zeros_like(esum,dtype='float32')
            
            t_energy_in = true_energy[used_events:used_events+npart]
            t_x_in = true_x[used_events:used_events+npart]
            t_y_in = true_y[used_events:used_events+npart]
            
            t_iselectron_in = isElectron[used_events:used_events+npart]
            t_isgamma_in = isGamma[used_events:used_events+npart]
            t_ispositron_in = isPositron[used_events:used_events+npart]
            
            for i in range(npart):
                t_energy[t_idx==float(i)] = t_energy_in[i]
                tot_true_e += t_energy_in[i]
                t_x[t_idx==float(i)] = t_x_in[i]
                t_y[t_idx==float(i)] = t_y_in[i]
                
                t_iselectron[t_idx==float(i)] = t_iselectron_in[i]
                t_isgamma[t_idx==float(i)] = t_isgamma_in[i]
                t_ispositron[t_idx==float(i)] = t_ispositron_in[i]
            
            
            i_truth = np.concatenate([
                np.expand_dims(t_mask, axis=1),
                np.expand_dims(t_energy, axis=1),
                np.expand_dims(t_x, axis=1),
                np.expand_dims(t_y, axis=1),
                np.expand_dims(t_iselectron, axis=1),
                np.expand_dims(t_isgamma, axis=1),
                np.expand_dims(t_idx, axis=1),
                i_rhx, 
                i_rhy, 
                i_rhz,
                i_rhid
                                   ],
                
                axis=-1
                )
            
            i_feat = np.expand_dims(i_feat,axis=0)
            i_truth = np.expand_dims(i_truth,axis=0)
            
            used_events += npart
            feat.append(i_feat)
            truth.append(i_truth)
            layer.append(np.expand_dims(rechit_layer[0],axis=0))
            npart_arr.append([npart])
            truth_en.append([tot_true_e])
            
            
        feat = np.concatenate(feat,axis=0)
        truth = np.concatenate(truth,axis=0)
        layer = np.concatenate(layer,axis=0)
        
        npart_arr = np.concatenate(npart_arr,axis=0)
        truth_en = np.concatenate(truth_en,axis=0)
        
        return feat, np.array(truth,dtype='float32'), layer, npart_arr, truth_en
        
           
        '''
        outdict['t_mask'] =  tf.reshape(truth[:,:,:,0:1], resh(1)) 
    outdict['t_pos']  =  tf.reshape(truth[:,:,:,1:3], resh(2), name="lala")/16.
    outdict['t_ID']   =  tf.reshape(truth[:,:,:,3:6], resh(3))  
    outdict['t_dim']  =  tf.reshape(truth[:,:,:,6:8], resh(2))/4.
    outdict['t_objidx']= tf.reshape(truth[:,:,:,8:9], resh(1))

        '''
        
        
        #create truth index:
        
    
    def _convertFromSourceFile(self, filename, weighterobjects, istraining):
    
        fileTimeOut(filename, 10)#10 seconds for eos to recover 
        
        tree = uproot.open(filename)["B4"]
        nevents = tree.numentries
        
        #truth
        isElectron  = self.tonumpy(tree["isElectron"].array()   )
        isGamma     = self.tonumpy(tree["isGamma"].array()      )
        isPositron  = self.tonumpy(tree["isPositron"].array()   )
        true_energy = self.tonumpy(tree["true_energy"].array()  )
        true_x = self.tonumpy(tree["true_x"].array()  )
        true_y = self.tonumpy(tree["true_y"].array()  )
        
        rechit_energy = self.tonumpy(tree["rechit_energy"].array() )
        rechit_x      = self.tonumpy(tree["rechit_x"].array()      )
        rechit_y      = self.tonumpy(tree["rechit_y"].array()      )
        rechit_z      = self.tonumpy(tree["rechit_z"].array()      )
        rechit_layer  = self.tonumpy(tree["rechit_layer"].array()  )
        rechit_detid  = self.tonumpy(tree['rechit_detid'].array()   )
        
        #print('rechit_energy',rechit_energy,rechit_energy.shape)
        #print(rechit_detid)
        
        #for ...
        feat, truth, layers, npart_arr, truth_en = self.mergeShowers( 
                     isElectron,
                     isGamma,
                     isPositron,
                     true_energy,
                     true_x,
                     true_y,
                     
                     rechit_energy,
                     rechit_x,
                     rechit_y,
                     rechit_z,
                     rechit_layer,
                     rechit_detid,
                     
                     maxpart=self.npart,
                     istraining=istraining)
        
        
        print('feat',feat.shape)
        
        calo,track = self.separateLayers(feat,layers)
        
        calo = np.reshape(calo, [truth.shape[0],16,16,-1])
        track = np.reshape(track, [truth.shape[0],64,64,-1])
        
        #print(calo[0,:,:,1:3])
        #this needs to be rebinned in x and y
        #calosort = np.argsort(calo[:,:,1]*100+calo[:,:,2], axis=-1)
        #calo = calo[calosort]
        #calo = np.reshape(calo, [truth.shape[0],16,16,-1])
        
        debug=False
        if debug:
            import matplotlib.pyplot as plt
            calotruth = truth[:,0:16*16,:]
            calotruth = np.reshape(calotruth, [truth.shape[0],16,16,-1])
            tracktruth = truth[:,16*16:,:]
            tracktruth = np.reshape(tracktruth, [truth.shape[0],64,64,-1])
            for event in range(10):
                #print truth index and rec energy
                fig,ax =  plt.subplots(1,1)
                ax.imshow(calotruth[event,:,:,6], aspect=1)
                fig.savefig("calo_idx"+str(event)+".pdf")
                ax.imshow(calo[event,:,:,0], aspect=1)
                fig.savefig("calo_en"+str(event)+".pdf")
                ax.imshow(calotruth[event,:,:,0], aspect=1)
                fig.savefig("calo_tmask"+str(event)+".pdf")
                
                ax.imshow(tracktruth[event,:,:,6], aspect=1)
                fig.savefig("tracktruth_idx"+str(event)+".pdf")
                ax.imshow(track[event,:,:,0], aspect=1)
                fig.savefig("track_en"+str(event)+".pdf")
                ax.imshow(tracktruth[event,:,:,0], aspect=1)
                fig.savefig("tracktruth_tmask"+str(event)+".pdf")
                
            
        print('calo',calo.shape)
        print('track',track.shape)
        print('truth',truth.shape)
        
        #np.set_printoptions(threshold=10*1280)
        
        #print('calo',calo[0])
        #print('track',track[0])
        
        if hasattr(self, "truth_en"):
            self.truth_en=truth_en
        
        return [calo,track],[truth],[]#[tracker0, tracker1, tracker2, tracker3, calo] , [trutharray], []

    def convertFromSourceFile(self, filename, weighterobjects, istraining):
        f,t,w = self._convertFromSourceFile(filename, weighterobjects, istraining)
        return f,t,w

    def writeOutPrediction(self, predicted, features, truth, weights, outfilename, inputfile):
        # predicted will be a list
        print('writeout')
        print('predicted',predicted[0].shape)
        print('features',features[0].shape)
        print('truth',truth[0].shape)
        
        
        parr = predicted[0] #unroll(predicted[0])
        farr = features[0] #unroll(features[0])
        tarr = truth[0] #unroll(truth[0])
        
        from DeepJetCore.TrainData import TrainData
        #use traindata as data storage
        td = TrainData()
        td._store([parr, farr, tarr],[],[])
        td.writeToFile(outfilename)
        
        
    
class TrainData_PF_graph(TrainData_PF):
    def __init__(self):
        TrainData_PF.__init__(self)
        
        
    def convertFromSourceFile(self, filename, weighterobjects, istraining):
        f,t,_ = TrainData_PF._convertFromSourceFile(self, filename, weighterobjects, istraining)
        
        for i  in range(2):
            f[i] = np.reshape(f[i],[f[i].shape[0], -1, f[i].shape[-1]]) #flatten
        
        f = np.concatenate([f[0],f[1]],axis=1)
        
        
        print('f shaped',f.shape)
        
        t = np.reshape(t[0],[t[0].shape[0], -1, t[0].shape[-1]]) #flatten
        
        #sort by energy
        en_sort = np.argsort(-f[:,:,0:1], axis=1)
        print('en_sort',en_sort.shape)
        f = np.take_along_axis(f, en_sort, axis=1)
        t = np.take_along_axis(t, en_sort, axis=1)
        
        f = f[:,:200,:]
        t = t[:,:200,:]
        
        
        
        return [np.array(f,dtype='float32')],[np.array(t,dtype='float32')],[]
        
        

        
class TrainData_PF_hipart(TrainData_PF):
    def __init__(self):
        TrainData_PF.__init__(self)
        self.npart=15    

class TrainData_PF_graph_hipart(TrainData_PF_graph): 
    def __init__(self):
        TrainData_PF_graph.__init__(self)
        self.npart=15   
        
#for calibration
class TrainData_PF_onepart(TrainData_PF):
    def __init__(self):
        TrainData_PF.__init__(self)
        self.npart=1   
        
        
        
class TrainData_PF_jet(TrainData_PF): 
    def __init__(self):
        TrainData_PF.__init__(self)
        self.npart=-15 #produce jets  
        
        
class TrainData_PF_graph_jet(TrainData_PF_graph): 
    def __init__(self):
        TrainData_PF_graph.__init__(self)
        self.npart=-15 #produce jets  
        
         
        
    