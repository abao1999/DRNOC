

###
#
#
# for testing: rm -rf TEST; python gravnet.py /eos/cms/store/cmst3/group/hgcal/CMG_studies/gvonsem/hgcalsim/ConverterTask/closeby_1.0To100.0_idsmix_dR0.1_n10_rnd1_s1/dev_LayerClusters_prod2/testconv/dataCollection.dc TEST
#
###

import DeepJetCore
from DeepJetCore.training.training_base import training_base
from DeepJetCore.DataCollection import DataCollection
import keras
from keras.models import Model
from keras.layers import  Reshape, Dense,Conv1D, Conv2D, BatchNormalization, Multiply, Concatenate, Dropout,MaxPooling2D, UpSampling2D #etc
from Layers import Conv2DGlobalExchange, PadTracker, CropTracker, TileCalo, GaussActivation, Tile2D, TileTrackerFeatures
from DeepJetCore.DJCLayers import ScalarMultiply, Clip, SelectFeatures, Print

from tools import plot_pred_during_training, plot_truth_pred_plus_coords_during_training, plot_particle_resolution_during_training
import tensorflow as tf
import os

from Layers import GravNet_simple, GlobalExchange
from Losses import particle_condensation_loss,dummy

nbatch=550#120 #1*7

plots_after_n_batch=1 #1000
use_event=0
learningrate=3e-4 #-4

momentum=0.6


def output_block(x,ids,energy_raw):
    p_beta    = Dense(1,activation='sigmoid')(x)
    p_tpos    = ScalarMultiply(10.)(Dense(2)(x))
    p_ID      = Dense(2,activation='softmax')(x)

    p_E       = (Dense(1)(x))
    p_ccoords = ScalarMultiply(10.)(Dense(2)(x))
    
    predictions=Concatenate()([p_beta , 
                               p_E    , 
                               p_tpos   ,
                               p_ID     ,
                               p_ccoords,
                               ids,
                               energy_raw])
    
    print('predictions',predictions.shape)
    return predictions
    
def checkids(Inputs):
    
    
    return SelectFeatures(5,6)(Inputs[0])
    

    
def minimodel(Inputs,feature_dropout=-1.):
    x = Inputs[0] #this is the self.x list from the TrainData data structure
    energy_raw = SelectFeatures(0,3)(x)
    x = BatchNormalization(momentum=0.6)(x)
    
    feat=[x]
    
    for i in range(6):
        #add global exchange and another dense here
        x = GlobalExchange()(x)
        x = Dense(64, activation='elu')(x)
        x = Dense(64, activation='elu')(x)
        x = BatchNormalization(momentum=0.6)(x)
        x = Dense(64, activation='elu')(x)
        x = GravNet_simple(n_neighbours=10, 
                 n_dimensions=4, 
                 n_filters=128, 
                 n_propagate=64)(x)
        x = BatchNormalization(momentum=0.6)(x)
        feat.append(Dense(32, activation='elu')(x))
    
    x = Concatenate()(feat)
    x = Dense(64, activation='elu')(x)

    return Model(inputs=Inputs, outputs=output_block(x,checkids(Inputs),energy_raw))
    
  


train=training_base(testrun=False,resumeSilently=True,renewtokens=False)

import os
os.system('cp /storage/user/abao/abao/SOR/modules/betaLosses.py '+train.outputDir+'/')


from tools import plot_pixel_3D_clustering_flat_during_training_graph

#samplepath = "/data/hgcal-0/store/jkiesele/SOR/Dataset/test_wiggle/100.djctd"
#samplepath = train.val_data.getSamplePath(train.val_data.samples[0])
samplepath = "/storage/user/abao/abao/SOR/data/test_data/9.djctd"

def decay_function(ncalls):
    #print('call decay')
    #return 500
    if ncalls > 1000:
        return 500
    if ncalls > 200:
        return 50
    if ncalls > 100:
        return 20
    return 10


def reso_decay_function(ncalls):
    #print('call decay')
    #return 500
    if ncalls > 1000:
        return 3000
    if ncalls > 200:
        return 1000
    if ncalls > 100:
        return 50
    return 30

#only plots calo but fine
ppdts= [plot_pixel_3D_clustering_flat_during_training_graph(
               samplefile=samplepath,
               output_file=train.outputDir+'/train_progress'+str(i),
               use_event=use_event+i,
               afternbatches=plots_after_n_batch,
               on_epoch_end=False,
               mask=False,
               ccorrdsx_idx=6,
               ccorrdsy_idx=7,
               #cut_truth=16*16,
               assoindex=6,
               feat_x=1,
               feat_y=3,
               feat_z=2,
               decay_function=decay_function
               ) for i in range(5) ]

resoplot = plot_particle_resolution_during_training(
    outfilename=train.outputDir+'/resolution',
    samplefile=samplepath,
    after_n_batches=1,
    decay_function=reso_decay_function,
    use_event=-1
    )

if not train.modelSet(): # allows to resume a stopped/killed training. Only sets the model if it cannot be loaded from previous snapshot

    #for regression use the regression model
    train.setModel(minimodel)#model)
    
    #read weights where possible from pretrained model
    #import os
    #from DeepJetCore.modeltools import load_model, apply_weights_where_possible
    #m_weights =load_model(os.environ['DEEPJETCORE_SUBPACKAGE'] + '/pretrained/gravnet_1.h5')
    #train.keras_model = apply_weights_where_possible(train.keras_model, m_weights)
    
    #for regression use a different loss, e.g. mean_squared_error
train.compileModel(learningrate=learningrate,
                   #loss=dummy,
                   loss=particle_condensation_loss,
                   #clipnorm=1
                   )#metrics=[pixel_over_threshold_accuracy]) 
                  
print(train.keras_model.summary())
#exit()

ppdts_callbacks=[ppdts[i].callback for i in range(len(ppdts))]
ppdts_callbacks.append(resoplot)

verbosity=2

#train.change_learning_rate(learningrate/10.)
model,history = train.trainModel(nepochs=20, 
                                 batchsize=int(nbatch),
                                 checkperiod=10, # saves a checkpoint model every N epochs
                                 verbose=verbosity,
                                 additional_callbacks=ppdts_callbacks)
print('reducing learning rate')
train.change_learning_rate(learningrate/10.)

model,history = train.trainModel(nepochs=100+20, 
                                 batchsize=nbatch,
                                 checkperiod=10, # saves a checkpoint model every N epochs
                                 verbose=verbosity,
                                 additional_callbacks=ppdts_callbacks)

print('reducing learning rate')
train.change_learning_rate(learningrate/10.)

model,history = train.trainModel(nepochs=200, 
                                 batchsize=nbatch,
                                 checkperiod=10, # saves a checkpoint model every N epochs
                                 verbose=verbosity,
                                 additional_callbacks=ppdts_callbacks)


