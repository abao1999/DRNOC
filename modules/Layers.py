
global_layers_list = {}

from keras.layers import Layer
import keras.backend as K
import tensorflow as tf

from caloGraphNN import *
from caloGraphNN_keras import GlobalExchange
global_layers_list['GlobalExchange']=GlobalExchange  

# Define custom layers here and add them to the global_layers_list dict (important!)


class Conv2DGlobalExchange(Layer):
    '''
    Centers phi to the first input vertex, such that the 2pi modulo behaviour 
    disappears for a small selection
    '''
    def __init__(self, **kwargs):
        super(Conv2DGlobalExchange, self).__init__(**kwargs)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],input_shape[2],input_shape[3]+input_shape[3])
    
    def call(self, inputs):
        average = tf.reduce_mean(inputs, axis=[1,2], keepdims=True)
        average = tf.tile(average, [1,tf.shape(inputs)[1],tf.shape(inputs)[2],1])
        return tf.concat([inputs,average],axis=-1)
        
    
    def get_config(self):
        base_config = super(Conv2DGlobalExchange, self).get_config()
        return dict(list(base_config.items()))
        

global_layers_list['Conv2DGlobalExchange']=Conv2DGlobalExchange 


class PadTracker(Layer):
    '''
    Centers phi to the first input vertex, such that the 2pi modulo behaviour 
    disappears for a small selection
    '''
    def __init__(self, **kwargs):
        super(PadTracker, self).__init__(**kwargs)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0],2*32,2*32,input_shape[3])
    
    def call(self, inputs):
        return tf.pad(inputs, [[0,0],[16,16],[16,16],[0,0]])
        
    
    def get_config(self):
        base_config = super(PadTracker, self).get_config()
        return dict(list(base_config.items()))
        

global_layers_list['PadTracker']=PadTracker 


class CropTracker(Layer):
    '''
    Centers phi to the first input vertex, such that the 2pi modulo behaviour 
    disappears for a small selection
    '''
    def __init__(self, **kwargs):
        super(CropTracker, self).__init__(**kwargs)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0],32,32,input_shape[3])
    
    def call(self, inputs):
        return inputs[:,16:48,16:48,:]
        
    
    def get_config(self):
        base_config = super(CropTracker, self).get_config()
        return dict(list(base_config.items()))
        

global_layers_list['CropTracker']=CropTracker 


class TileTrackerFeatures(Layer):
    '''
    Centers phi to the first input vertex, such that the 2pi modulo behaviour 
    disappears for a small selection
    '''
    def __init__(self, **kwargs):
        super(TileTrackerFeatures, self).__init__(**kwargs)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],input_shape[2],input_shape[3]*4)
    
    def call(self, inputs):
        return tf.tile(inputs, [1,1,1,4])
        
    
    def get_config(self):
        base_config = super(TileTrackerFeatures, self).get_config()
        return dict(list(base_config.items()))
        

global_layers_list['TileTrackerFeatures']=TileTrackerFeatures 



class TileCalo(Layer):
    '''
    Centers phi to the first input vertex, such that the 2pi modulo behaviour 
    disappears for a small selection
    '''
    def __init__(self, **kwargs):
        super(TileCalo, self).__init__(**kwargs)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0],4*16,4*16,input_shape[3])
    
    def call(self, inputs):
        return tf.tile(inputs, [1,4,4,1])
        
    
    def get_config(self):
        base_config = super(TileCalo, self).get_config()
        return dict(list(base_config.items()))
        

global_layers_list['TileCalo']=TileCalo 

class Tile2D(Layer):
    '''
    Centers phi to the first input vertex, such that the 2pi modulo behaviour 
    disappears for a small selection
    '''
    def __init__(self, ntiles, **kwargs):
        super(Tile2D, self).__init__(**kwargs)
        self.ntiles=ntiles
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0],4*16,4*16,input_shape[3])
    
    def call(self, inputs):
        return tf.tile(inputs, [1,self.ntiles,self.ntiles,1])
        
    
    def get_config(self):
        base_config = super(Tile2D, self).get_config()
        config = {'ntiles' :self.ntiles}
        return dict(list(config.items() ) + list(base_config.items()) )
        

global_layers_list['Tile2D']=Tile2D 


class GaussActivation(Layer):
    '''
    Centers phi to the first input vertex, such that the 2pi modulo behaviour 
    disappears for a small selection
    '''
    def __init__(self, **kwargs):
        super(GaussActivation, self).__init__(**kwargs)
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def call(self, inputs):
        return tf.exp(- inputs**2 )
        
    
    def get_config(self):
        base_config = super(GaussActivation, self).get_config()
        return dict(list(base_config.items()) )
        

global_layers_list['GaussActivation']=GaussActivation 






class GravNet_simple(tf.keras.layers.Layer):
    def __init__(self, 
                 n_neighbours, 
                 n_dimensions, 
                 n_filters, 
                 n_propagate,**kwargs):
        super(GravNet_simple, self).__init__(**kwargs)
        
        self.n_neighbours = n_neighbours
        self.n_dimensions = n_dimensions
        self.n_filters = n_filters
        self.n_propagate = n_propagate

        with tf.name_scope(self.name+"/1/"):
            self.input_feature_transform = tf.keras.layers.Dense(n_propagate)
        with tf.name_scope(self.name+"/2/"):
            self.input_spatial_transform = tf.keras.layers.Dense(n_dimensions)
        with tf.name_scope(self.name+"/3/"):
            self.output_feature_transform = tf.keras.layers.Dense(n_filters, activation='tanh')

    def build(self, input_shape):
        with tf.name_scope(self.name+"/1/"):
            self.input_feature_transform.build(input_shape)
        with tf.name_scope(self.name+"/2/"):
            self.input_spatial_transform.build(input_shape)
        with tf.name_scope(self.name+"/3/"):
            self.output_feature_transform.build((input_shape[0], input_shape[1],
                                                 input_shape[2] + self.input_feature_transform.units * 2))
 
        super(GravNet_simple, self).build(input_shape)
        
    def call(self, x):
        
        coordinates = self.input_spatial_transform(x)
        features = self.input_feature_transform(x)
        collected_neighbours = self.collect_neighbours(coordinates, features)
        
        updated_features = tf.concat([x, collected_neighbours], axis=-1)
        return self.output_feature_transform(updated_features)
    

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_feature_transform.units)
    
    def collect_neighbours(self, coordinates, features):
        
        distance_matrix = euclidean_squared(coordinates, coordinates)
        ranked_distances, ranked_indices = tf.nn.top_k(-distance_matrix, self.n_neighbours)

        neighbour_indices = ranked_indices[:, :, 1:]
        
        n_batches = tf.shape(features)[0]
        n_vertices = tf.shape(features)[1]
        n_features = tf.shape(features)[2]
        
        batch_range = tf.range(0, n_batches)
        batch_range = tf.expand_dims(batch_range, axis=1)
        batch_range = tf.expand_dims(batch_range, axis=1)
        batch_range = tf.expand_dims(batch_range, axis=1) # (B, 1, 1, 1)

        # tf.ragged FIXME? n_vertices
        batch_indices = tf.tile(batch_range, [1, n_vertices, self.n_neighbours - 1, 1]) # (B, V, N-1, 1)
        vertex_indices = tf.expand_dims(neighbour_indices, axis=3) # (B, V, N-1, 1)
        indices = tf.concat([batch_indices, vertex_indices], axis=-1)
    
    
        distance = -ranked_distances[:, :, 1:]
    
        weights = gauss_of_lin(distance * 10.)
        weights = tf.expand_dims(weights, axis=-1)
        
        neighbour_features = tf.gather_nd(features, indices)
        neighbour_features *= weights
        neighbours_max = tf.reduce_max(neighbour_features, axis=2)
        neighbours_mean = tf.reduce_mean(neighbour_features, axis=2)
        
        return tf.concat([neighbours_max, neighbours_mean], axis=-1)
    
    def get_config(self):
        config = {'n_neighbours': self.n_neighbours, 
                  'n_dimensions': self.n_dimensions, 
                  'n_filters': self.n_filters, 
                  'n_propagate': self.n_propagate}
        base_config = super(GravNet_simple, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



global_layers_list['GravNet_simple']=GravNet_simple   






