

#
#
# 2 inputs: one calo, one tracker
# calo:    16x16
# tracker: 64x64
#
# use trainData as input. feat[0] and feat[1]
#
# inputs are: energy, x, y
#
#



# standard PF as in paper: one track per ECal cluster, discard rest
#
#
# Here: also check other tracks and split cluster multiple times if needed
#
#
#







import numpy as np
from numba import jit

import math


E_seed = 0.230
E_cell = 0.080
sigma = 15 #(everything mm)
cellsize = 22 #(everything mm)

class calo_cluster(object):
    def __init__(self,energy,pos,true_energy=-1,true_pos=[-1000,-1000],auto_correct=True):
        #self.seed_idx=[]
        self.raw_energy=float(energy)
        if auto_correct:
            self.energy = self.corrected_energy()
        else:
            self.energy =float(energy)
        self.position=pos
        self.true_energy=float(true_energy)
        self.true_position=true_pos
        
    def __str__(self):
        outstr="CaloCluster: energy "+str(self.energy)+"+-"+str(self.rel_resolution()*self.energy)+" GeV, pos "+str(self.position)
        outstr+=" truth energy " +str(self.true_energy)+" truth pos "+str(self.true_position)
        return outstr
    
    def rel_resolution(self):
        a = 0.028/math.sqrt(self.raw_energy)
        b = 0.12/self.raw_energy
        c = 0.003
        return math.sqrt(a**2 +b**2 + c**2)
    
    def corrected_energy(self):
        correction=[
0.241441, 1.20361, 1.14461, 1.12521, 1.11285, 1.10452, 1.11832, 1.0865, 1.08351, 1.08099, 1.09204, 1.07069, 1.07082, 1.06771, 1.06573, 1.06121, 1.05735, 1.05854, 1.05934, 1.05747, 1.05394, 1.05192, 1.05106, 1.04912, 1.07047, 1.04909, 1.04826, 1.06585, 1.04463, 1.04564, 1.04618, 1.04536, 1.04345, 1.04154, 1.0445, 1.04161, 1.04097, 1.0425, 1.07613, 1.05833, 1.04026, 1.06201, 1.03874, 1.03857, 1.03634, 1.0397, 1.03757, 1.03706, 1.03889, 1.05199, 1.03512, 1.04224, 1.03663, 1.03828, 1.03725, 1.06023, 1.03409, 1.03399, 1.05777, 1.04994, 1.03729, 1.03663, 1.03354, 1.03337, 1.03305, 1.04891, 1.03328, 1.03271, 1.03264, 1.03334, 1.03101, 1.03255, 1.0316, 1.03378, 1.03395, 1.03284, 1.03233, 1.03426, 1.03195, 1.03103, 1.03155, 1.03036, 1.03078, 1.03104, 1.03154, 1.0312, 1.05401, 1.05368, 1.03004, 1.03047, 1.03065, 1.02894, 1.02954, 1.03007, 1.02874, 1.03251, 1.03367, 1.02828, 1.02975, 1.03024, 1.02941, 1.02992, 1.02969, 1.03007, 1.02855, 1.02827, 1.0296, 1.03031, 1.03056, 1.02804, 1.0294, 1.02782, 1.02903, 1.02851, 1.03573, 1.02849, 1.02709, 1.03095, 1.02933, 1.02811, 1.02901, 1.02752, 1.02936, 1.03163, 1.03065, 1.02689, 1.02717, 1.02843, 1.02724, 1.02909, 1.02923, 1.02675, 1.03151, 1.02869, 1.02762, 1.03045, 1.0261, 1.03031, 1.0288, 1.03136, 1.02691, 1.02706, 1.02746, 1.02729, 1.0289, 1.02693, 1.02657, 1.02718, 1.02812, 1.02573, 1.0287, 1.02816, 1.02991, 1.0276, 1.0252, 1.02728, 1.02851, 1.0294, 1.02817, 1.02876, 1.02788, 1.03036, 1.02728, 1.02967, 1.02863, 1.02644, 1.02654, 1.02875, 1.02815, 1.02889, 1.02756, 1.02505, 1.02828, 1.02725, 1.02947, 1.02613, 1.02951, 1.02708, 1.02557, 1.0281, 1.02917, 1.02756, 1.02739, 1.02635, 1.02673, 1.0267, 1.02543, 1.02679, 1.02548, 1.0249, 1.02613, 1.02654, 1.02548, 1.02462, 1.02269, 1.02074, 1.01759, 1.01759, 1.01759, 1.01759, ]
        bin = int(self.raw_energy)
        correction2=[0.107671, 0.980985, 0.986455, 0.99232, 0.99543, 0.998903, 1.01679, 0.993694, 0.997247, 0.995484, 1.0125, 0.996244, 0.994529, 0.997132, 0.99958, 0.99719, 0.997252, 0.990351, 0.996097, 0.999822, 0.998417, 0.997762, 0.996421, 0.998102, 1.01792, 0.999546, 1.00029, 1.01615, 0.997746, 1.00032, 0.999011, 0.999285, 0.999075, 0.995929, 0.999717, 0.999518, 0.995177, 1.00028, 1.0315, 1.01436, 0.998965, 1.0186, 0.998517, 0.998775, 0.996494, 1.00099, 0.998829, 0.998056, 1.00056, 1.01235, 0.998597, 1.00552, 0.999183, 1.00012, 0.999487, 1.0221, 0.998161, 0.996597, 1.01573, 1.00995, 1.00057, 1.00167, 0.998119, 0.996794, 0.996942, 1.01446, 0.999717, 0.997411, 0.998252, 0.999343, 0.998163, 1.00031, 0.997525, 1.00054, 1.00093, 0.9997, 0.999632, 1.00105, 0.997707, 0.998323, 0.997013, 0.997935, 0.997884, 0.997408, 1.00032, 0.998061, 1.02072, 1.02091, 0.998388, 1.00001, 0.999792, 0.998847, 0.999723, 0.999653, 0.997464, 1.00078, 1.00299, 0.997675, 0.999403, 1.00019, 0.999772, 1.00016, 1.00033, 1.00031, 0.999272, 0.997788, 1.00026, 1.0002, 1.00063, 0.998279, 1.00038, 0.998734, 0.99978, 0.99835, 1.00684, 0.99883, 0.999351, 1.00189, 0.999547, 0.998593, 1.00046, 0.99774, 1.00057, 1.00258, 1.00083, 0.998352, 0.998333, 1.00026, 0.999467, 1.00128, 1.00113, 0.998262, 1.00243, 1.00003, 1.00012, 1.0022, 0.99821, 1.00085, 0.999645, 1.00283, 0.998917, 0.99881, 0.999703, 0.999171, 1.00149, 0.999903, 0.998544, 0.999487, 1.00046, 0.99771, 1.00103, 0.99981, 1.00211, 0.999834, 0.997281, 0.999629, 1.00152, 1.00114, 1.00032, 1.00054, 0.999993, 1.00233, 0.999705, 1.00162, 1.00132, 0.998683, 0.99873, 1.00099, 0.999608, 1.00154, 0.999886, 0.998082, 1.00119, 1.00015, 1.00296, 0.999116, 1.0023, 1.0002, 0.99855, 1.00081, 1.00174, 1.00064, 0.999804, 0.999565, 1.00055, 1.00004, 0.998668, 1.0001, 0.999379, 0.999337, 1.00063, 1.00123, 1.00078, 1.00034, 1.00012, 1.00025, 0.999739,0.999739,0.999739,0.999739]
        if bin >= 200: 
            bin=199
        elif bin < 0:
            bin=0
        return  self.raw_energy*(correction[bin]/correction2[bin]) #+ 7.2063e-06*self.energy**3


class pf_track(object):
    def __init__(self,energy,pos,track_idx):
        self.energy=energy
        self.position=pos
        self.track_idx=track_idx
        
    def rel_resolution(self):
        pt=self.energy
        return (pt/100.)*(pt/100.)*0.04 +0.01;


class pfCandidate(calo_cluster):
    def __init__(self,energy=-1,pos=[0.,0.]):
        calo_cluster.__init__(self,energy,pos, auto_correct=False)
        self.track_idx=-1
    
    def create_from_link(self, calocluster, pftrack=None):
        if pftrack is None:
            self.energy=calocluster.energy
            self.position=calocluster.position
            return None
        else: #use a weighted mean
            
            self.position = 1./(4+1)*(4.*pftrack.position+calocluster.position)
            self.track_idx = pftrack.track_idx
            t = pftrack.energy
            dt = pftrack.rel_resolution()
            c = calocluster.energy
            dc = calocluster.rel_resolution()
            
            combined_error=math.sqrt((c*dc)**2+(t*dt)**2)
            
            if c-t > combined_error: #incompatible, create second pf candidate
                self.energy = t #use track momentum
                self.position = pftrack.position
                neutral_cand = pfCandidate(c-t, calocluster.position)
                if c-t > 0.5:
                    return neutral_cand
            elif t-c > combined_error: #larger track momentum
                self.energy=calocluster.energy
                self.position=calocluster.position
                return None
                
            #what remains is if they are compatible, so here we are
            self.energy = 1/((t*dt)**(-2) + (c*dc)**(-2)) * (c/(c*dc)**2 +t/(t*dt)**2)
            
            return None
        


@jit(nopython=True)     
def gen_get_truth_particles(eventtruth):#truth input: V x F
    truthpos    =eventtruth[:,2:4]
    truthenergy = eventtruth[:,1:2]
    tpos=[]
    ten =[]
    for i in range(truthpos.shape[0]):
        en = truthenergy[i][0]
        if en==0 or en in ten: continue
        pos = [truthpos[i][0],truthpos[i][1]]
        tpos.append(pos)
        ten.append(en)
    return np.array(tpos),np.array(ten)
     

#
# Add the tracks here in the sense that:
# If not seed, but track in front (within 2.2/2 cm) -> make seed
#
@jit(nopython=True)     
def c_calo_getSeeds(caloinput,seed_idxs,trackerinput,map):
    #get tracker hits first
    # and mark as seeds whatever calo is behind
    #make the mask:
    trackgrid_position = trackerinput[0,:,:,1:3]#positions
    calogrid_position = caloinput[0,:,:,1:3]#positions
    #make a map between tracker and calo
    for xc in range(calogrid_position.shape[0]):
        for yc in range(calogrid_position.shape[1]):
            for xt in range(trackgrid_position.shape[0]):
                for yt in range(trackgrid_position.shape[1]):
                    diffx = abs(trackgrid_position[xt,yt,0]-calogrid_position[xc,yc,0])
                    diffy = abs(trackgrid_position[xt,yt,1]-calogrid_position[xc,yc,1])
                    if diffx < 22/2. and diffy < 22/2.:
                        map[xt,yt,0]=xc
                        map[xt,yt,1]=yc
                    
            
    
    
    for ev in range(trackerinput.shape[0]):
        for xt in range(trackerinput.shape[1]):
            for yt in range(trackerinput.shape[2]):
                if trackerinput[ev,xt,yt,0] > 0.1:
                    seed_idxs[ev,map[xt,yt,0],map[xt,yt,1]]=1
        
        
    
    
    
    xmax=caloinput.shape[1]
    ymax=xmax #symmetric
    for ev in range(len(caloinput)):
        for x in range(len(caloinput[ev])):
            for y in range(len(caloinput[ev][x])):
                seed_e = caloinput[ev,x,y,0]
                if seed_e < E_seed:
                    continue
                #check surrounding cells
                is_seed=True
                for xs in range(x-1,x+2):
                    if xs<0 or xs>=xmax: continue
                    for ys in range(y-1,y+2):
                        if ys<0 or ys>=ymax: continue
                        if xs == x and ys ==y: continue
                        if caloinput[ev,xs,ys,0]>=seed_e:
                            is_seed = False
                        if not is_seed: break
                    if not is_seed: break
                if is_seed:
                    seed_idxs[ev,x,y]=1
    outseedidxs=[]
    for ev in range(len(caloinput)):
        outseedidxs.append(seed_idxs[ev]>0)
    return outseedidxs, seed_idxs

def calo_getSeeds(caloinput,trackerinput):
    seed_idxs=np.zeros( [caloinput.shape[0],caloinput.shape[1],caloinput.shape[2]] )
    map = np.zeros((trackerinput.shape[1],trackerinput.shape[2],2), dtype='int64')-1
    sidxl, seedids = c_calo_getSeeds(caloinput,seed_idxs,trackerinput,map)
    return sidxl, seedids


def calo_calc_f(A,c,mu):
    #expand i: aka the seed dim to axis0
    A_exp  = np.expand_dims(A, axis=0)  # 1 x N 
    mu_exp = np.expand_dims(mu, axis=0) # 1 x N x 2
    c_exp  = np.expand_dims(c, axis=1)  # M x 1 x 2
    
    posdelta = np.sum((c_exp-mu_exp)**2, axis=-1) # M x N
    
    upper = A_exp * np.exp(- posdelta/(2*sigma**2)) # M x N
    lower = np.sum( upper, axis=1, keepdims=True )
    
    return upper/(lower + 1e-7)
    
def calo_calc_A(f,E):# E: M , f: M x N
    E_exp = np.expand_dims(E, axis=1) # M x 1
    A = np.sum( f*E_exp , axis=0) # N
    return A
    
def calo_calc_mu(f,c,E): #c: M x 2, f: M x N, E: M 
    c_exp = np.expand_dims(c, axis=1) # M x 1 x 2
    f_exp = np.expand_dims(f, axis=2) # M x N x 1
    E_exp = np.expand_dims(np.expand_dims(E, axis=1), axis=1) # M x 1 x 1
    
    den = np.sum(f_exp*E_exp,axis=0)
    mu = np.sum(f_exp*E_exp*c_exp, axis=0) / den # N x 2
    return mu
    
    
def calo_determineClusters(caloinput, seedidxs):#calo input per event: x X y X F  /// seedidxs: also per event
    
    energies = np.reshape(caloinput[:,:,0:1], [-1,1]) #keep dim
    energies[energies<E_cell] = 0
    allhits = np.concatenate([energies, np.reshape(caloinput[:,:,1:], [caloinput.shape[0]**2, -1])],axis=-1)
    
    seed_properties = caloinput[seedidxs] #now 
    if len(seed_properties)<1:
        return []
    seed_energies = seed_properties[:,0]
    seed_pos = seed_properties[:,1:3]
    
    #make the matrices
    E = allhits[:,0]
    c = np.array(allhits[:,1:3]) 
    
    A = np.array(seed_energies) #initial
    mu = np.array(seed_pos) #initial
    not_converged=True
    f = calo_calc_f(A, c, mu)
    
    counter=0
    while(not_converged and counter<100):
        #use initial A for now
        new_mu = calo_calc_mu(f,c,E)
        mmudiff = np.max( np.sum((new_mu-mu)**2,axis=-1) )
        #print(mmudiff)
        if mmudiff < .2**2:
            not_converged = False
        mu = new_mu
        f = calo_calc_f(A, c, mu)
        counter+=1
    #print(counter)
    A = calo_calc_A(f,E)
    
    out=[]
    for i in range(len(A)):
        if not np.isnan(A[i]): #fit failed
            out.append(calo_cluster(A[i],mu[i]))
    
    return out
    

#@jit(nopython=True)     
def c_match_cluster_to_truth(clusters, true_pos, true_en, truth_used):# per event, true_pos: sequence. find truth for reco
    for cl in clusters:
        L1_dist=10000
        best_it=-1
        for i_t in range(len(true_pos)):
            if truth_used[i_t]: continue
            this_L1_dist = abs(cl.position[0]-true_pos[i_t][0])+abs(cl.position[1]-true_pos[i_t][1])
            if this_L1_dist < 2.*22. and this_L1_dist < L1_dist:
                 best_it=i_t
                 L1_dist=this_L1_dist
        if best_it>-1:
            truth_used[best_it]=1
            cl.true_energy = true_en[best_it]
            cl.true_position[0] = true_pos[best_it][0]
            cl.true_position[1] = true_pos[best_it][1]
    
    
    return clusters
    
def match_cluster_to_truth(clusters, true_pos, true_en):
    truth_used = np.zeros_like(true_en, dtype='int64')
    return c_match_cluster_to_truth(clusters, true_pos, true_en, truth_used)
    
    
def create_pf_tracks(tracks):
    #input tracks[event,:,:,:] 0: energy, 1,2 pos
    out_tracks = []
    intracks = np.reshape(tracks, [-1,tracks.shape[-1]])
    for i_t in range(len(intracks)):
        if intracks[i_t][0] < 1: continue
        out_tracks.append(pf_track(intracks[i_t][0],intracks[i_t,1:3],i_t+16*16))#calo offset
    return out_tracks

@jit(nopython=True)  
def c_perform_linking(cluster_positions, track_positions, track_matched, standardPF=False):
    
    matching=[] #i_cluster, i_track
    if standardPF:
        for i_c in range(len(cluster_positions)):
            best_distancesq = 1e3**2
            best_track = -1
            other_tracks=[]
            for i_t in range(len(track_positions)):
                distsq = (cluster_positions[i_c][0]-track_positions[i_t][0])**2 + (cluster_positions[i_c][1]-track_positions[i_t][1])**2
                if distsq > cellsize**2: continue
                other_tracks.append(i_t)
                if best_distancesq < distsq: continue
                best_track = i_t
                best_distancesq = distsq
            if best_track >=0 :
                track_matched[best_track] = True
            
            if best_track>=0:
                matching.append([i_c, best_track])
            else:
                matching.append([i_c])
   
    else:#match clusters to tracks
        for i_c in range(len(cluster_positions)):
            matching.append([i_c])
        for i_t in range(len(track_positions)):
            best_distancesq = 1e3**2
            best_cl = -1
            for i_c in range(len(cluster_positions)):
                distsq = (cluster_positions[i_c][0]-track_positions[i_t][0])**2 + (cluster_positions[i_c][1]-track_positions[i_t][1])**2
                if distsq > cellsize**2: continue
                if best_distancesq < distsq: continue
                best_distancesq = distsq
                best_cl=i_c
            if best_cl >=0:
                track_matched[i_t] = True
                matching[best_cl].append(i_t)
                
    return matching, track_matched
    
#works on an event by event basis
#clusters are calo_cluster objects, tracks are pf_track objects
#returns list of candidates directly
def perform_linking(clusters, tracks, standardPF=False):
    
    cluster_positions = np.array([c.position for c in clusters])
    track_positions = np.array([t.position for t in tracks])
    track_matched = np.array([False for i in range(len(tracks))])
    matching=[]
    if len(tracks):
        matching, track_matched = c_perform_linking(cluster_positions, track_positions, track_matched, standardPF)
    else:
        matching = [[i] for i in range(len(cluster_positions))]
        
    particles = []
    for m in matching:
        
        add_cands=[]
        cluster = clusters[m[0]]
        if len(m)>1:
            for i in range(1,len(m)):
                pfc = pfCandidate()
                if i > 1:
                    pass
                    #print('creating second part from cluster/track', cluster.energy, tracks[m[i]].energy)
                    #print('previous particle', particles[-1].energy)
                cand2 = pfc.create_from_link(cluster, tracks[m[i]])
                particles.append(pfc)
                if cand2 is None:
                    break
                cluster = cand2
                if standardPF:
                    particles.append(cand2)
                    break #only first track
                if i==len(m)-1:
                    particles.append(cand2)
                
                
                
        #if m[1]>=0:
        #    cand2 = pfc.create_from_link(cluster, tracks[m[1]])
        #    if cand2 is not None:
        #        if standardPF:
        #            add_cands.append(cand2)
        #        else: #split further
        #            for i in range(2,len(m)):
        #                
        else:
            pfc = pfCandidate()
            pfc.create_from_link(clusters[m[0]])
            particles.append(pfc)
            
            
    #non matched tracks
    if not standardPF:
        for i in range(len(track_matched)):
            if track_matched[i] : continue
            pfc = pfCandidate()
            pfc.create_from_link(tracks[i])
            particles.append(pfc)
            
    return particles
    
    
    
    
    

    





    