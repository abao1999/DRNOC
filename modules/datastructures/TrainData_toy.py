



from DeepJetCore.TrainData import TrainData, fileTimeOut
import numpy as np

class TrainData_toy(TrainData):
    def __init__(self):
        TrainData.__init__(self)

        
    
    def convertFromSourceFile(self, filename, weighterobjects, istraining):
    
        # this function defines how to convert the root ntuple to the training format
        # options are not yet described here
        from toygenerator import create_images
        
        seed = None
        if not istraining:
            seed = 100
            
        nevents=50
        if istraining:
            nevents=9000
        
        feature_array, trutharray = create_images(nevents,npixel=64,seed=seed,addwiggles=False)
        print('created', len(feature_array),' samples ')
        
        
        return [feature_array] , [trutharray], []


    def writeOutPrediction(self, predicted, features, truth, weights, outfilename, inputfile):
        # predicted will be a list
        print('writeout')
        print('predicted',predicted[0].shape)
        print('features',features[0].shape)
        print('truth',truth[0].shape)
        
        def unroll(a):
            a = np.reshape(a, [a.shape[0], a.shape[1]*a.shape[2], a.shape[3]])
            return a
        
        #unroll to event x vector
        # first 100 are enough for now
        parr = predicted[0][:100,...] #unroll(predicted[0])
        farr = features[0][:100,...] #unroll(features[0])
        tarr = truth[0][:100,...] #unroll(truth[0])
        
        from DeepJetCore.TrainData import TrainData
        #use traindata as data storage
        td = TrainData()
        td._store([parr, farr, tarr],[],[])
        td.writeToFile(outfilename)
    
    
        return
    
    
        from root_numpy import array2root
        out = np.core.records.fromarrays([parr[:,:,0], 
                                          parr[:,:,1], 
                                          parr[:,:,2], 
                                          parr[:,:,3], 
                                          parr[:,:,4], 
                                          parr[:,:,5], 
                                          parr[:,:,6], 
                                          parr[:,:,7], 
                                          parr[:,:,9], 
                                          parr[:,:,10], 
                                          
                                          tarr[:,:,0],
                                          tarr[:,:,1],
                                          tarr[:,:,2],
                                          tarr[:,:,3],
                                          tarr[:,:,4],
                                          tarr[:,:,5],
                                          tarr[:,:,6],
                                          tarr[:,:,7],
                                          
                                          farr[:,:,0],
                                          farr[:,:,1],
                                          farr[:,:,2],
                                          farr[:,:,3],
                                          farr[:,:,4],
                                          ],
                                             names='p_beta, p_posx, p_posy, p_ID0, p_ID1, p_ID2, p_dim1, p_dim2, p_ccoords1, p_coords2, t_mask, t_posx, t_posy, t_ID0, t_ID1, tID_2, t_dim1, t_dim2, f_r, f_g, f_b, f_x, f_y')
        
        array2root(out, outfilename, 'tree')
        
        
        
        '''
        outdict['t_mask'] =  tf.reshape(truth[:,:,:,0:1], reshaping) 
    outdict['t_pos']  =  tf.reshape(truth[:,:,:,1:3], reshaping, name="lala") 
    outdict['t_ID']   =  tf.reshape(truth[:,:,:,3:6], reshaping)  
    outdict['t_dim']  =  tf.reshape(truth[:,:,:,6:8], reshaping) 
    n_objects = truth[:,0,0,8]

    print('pred',pred.shape)

    outdict['p_beta']   =  tf.reshape(pred[:,:,:,0:1], reshaping)
    outdict['p_pos']    =  tf.reshape(pred[:,:,:,1:3], reshaping, name="lulu")
    outdict['p_ID']     =  tf.reshape(pred[:,:,:,3:6], reshaping)
    outdict['p_dim']    =  tf.reshape(pred[:,:,:,6:8], reshaping)
    p_object  = pred[:,0,0,8]
    outdict['p_ccoords'] = tf.reshape(pred[:,:,:,9:], reshaping)
        '''