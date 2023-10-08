from __future__ import print_function
import h5py
import numpy as np
import scipy.signal
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from time import time
import multiprocessing
import os
import matplotlib.pyplot as plt
from tensorflow_addons.optimizers import LAMB
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Dense, Activation, Dropout, Lambda, Multiply, Add, Concatenate, Flatten, Reshape, Softmax, GlobalMaxPool1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 

def normalize(strain):
    std = np.std(strain[:])
    strain[:] /= std
    return strain


# func. to make preds
def make_preds(whitened_L1, whitened_H1, whitened_V1, Model):

    # Load Strain
    normalized_L1 = normalize(whitened_L1)
    normalized_H1 = normalize(whitened_H1)
    normalized_V1 = normalize(whitened_V1)
    data = np.stack(( normalized_L1, normalized_H1,  normalized_V1), axis=1)

    # Create datagenerators
    dg_0 = TimeseriesGenerator(data=data, targets=data, length=4096, stride=4096, start_index=0, batch_size=256)
    dg_5 = TimeseriesGenerator(data=data, targets=data, length=4096, stride=4096, start_index=2047, batch_size=256)
    
    # Make preds
    preds_0 = Model.predict(dg_0, verbose=1)
    preds_5 = Model.predict(dg_5, verbose=1)
    
    return preds_0.ravel(), preds_5.ravel()


# func. to make preds
def mass_make_preds(whitened_L1, whitened_H1, whitened_V1, Model):

    # Load Strain
    normalized_L1 = normalize(whitened_L1)
    normalized_H1 = normalize(whitened_H1)
    normalized_V1 = normalize(whitened_V1)
    data = np.stack(( normalized_L1, normalized_H1,  normalized_V1), axis=1)

    # Create datagenerators
    dg_0 = TimeseriesGenerator(data=data, targets=data, length=4096, stride=4096, start_index=0, batch_size=256)
    
    # Make preds
    preds_0 = Model.predict_generator(dg_0, verbose=1)
    
    return preds_0.ravel()


# func. to find peaks
def find_peaks(preds, threshold = 0.9, width = [1500,3000], mean = 0.9):
    '''
    preds: 1D numpy array of sigmoid output from the NN
    '''
    test_p = preds
    
    peaks, properties =  scipy.signal.find_peaks(test_p, height=threshold, width = width, distance = 4096*1 )

    left = properties['left_ips']
    right = properties['right_ips']

    f_left = []
    f_right = []
    for i in range(len(left)):
        sliced = test_p[int(left[i]):int(right[i])] 
        if (np.mean(sliced>mean)>mean):
            f_left.append(int(left[i]))
            f_right.append(int(right[i]))
            
    return peaks, f_left, f_right


# Process with make preds and find peaks 
def Inference(spin, strain_L1, strain_H1, strain_V1):
    models = ['trained_models/model_spin_2.h5','trained_models/model_spin_3.h5','trained_models/model_spin_1.h5']
    string = 'Spin'

    detections = []

    for model in models:
        print(string, f'{len(models)} Model Ensemble: {models.index(model)+1}/{len(models)}')
        
        for threshold in [0.9999]:
            for width in [2000]:
                # Make preds
                Model = keras.models.load_model(model, custom_objects={'LAMB': LAMB})
                preds_0, preds_5 = make_preds(strain_L1, strain_H1, strain_V1, Model)

                preds_0 = preds_0.reshape(-1, 4096)
                preds_5 = preds_5.reshape(-1, 4096)

                detection_0 = []
                detection_5 = []
                pred_peak_0 = []
                pred_peak_5 = []
                for j in range(preds_0.shape[0]):
                    p_0 = preds_0[j]
                    peaks_0, f_left_0, f_right_0 =  find_peaks(p_0.flatten(), threshold=threshold, width=[width, 3000], mean=0.95)

                    if peaks_0.size != 0 :
                        detection_0.append(1)
                        pred_peak_0.append( f_right_0 )
                    else:
                        detection_0.append(0)
                        pred_peak_0.append(-1)

                for k in range(preds_5.shape[0]):
                    p_5 = preds_5[k]
                    peaks_5, f_left_5, f_right_5 =  find_peaks(p_5.flatten(), threshold=threshold, width=[width, 3000], mean=0.95)

                    if peaks_5.size != 0 :
                        detection_5.append(1)
                        pred_peak_5.append( f_right_5 )
                    else:
                        detection_5.append(0)
                        pred_peak_5.append(-1) 

        detection = np.concatenate((np.nonzero(detection_0)[0], np.nonzero(detection_5)[0]))
        detection = list(set(detection))
        detection = np.sort(detection)
        diff = np.diff(detection)
        indices = np.where(diff == 1)[0]
        detection = np.delete(detection, indices)
#         print(f'Model {models.index(model)+1} detection: ', detection)
        detections.append(detection)
    
    Pos= np.intersect1d(detections[0],detections[1])
    Pos= np.intersect1d(Pos,detections[2])
#     Pos= np.intersect1d(Pos,detections[3])
    print('Ensemble Detection: ', Pos)
            
    return detections, Pos


def Mass_Inference(Strain_H1, Strain_L1, Strain_V1, index):
    Models = ['trained_models/model_mass_1.h5','trained_models/model_mass_2.h5']
    predictions = []
    masses = [0]*len(index)
    for i in Models:
        Model = full_module(inp_shape = (4096,3))
        Model.load_weights(i)
        preds = mass_make_preds(Strain_L1, Strain_H1, Strain_V1, Model)
        for j,k in enumerate(index):  
            masses[j]+=preds[k]
    mass_predictions = [item / len(Models) for item in masses]
    for mass in mass_predictions:
        if mass>5:
            predictions.append(True)
        else:
            predictions.append(False)
    filtered_index = [num for num, b in zip(index, predictions) if b]
    print('Mass Filtered Positives:', filtered_index)
    return filtered_index


def Inference_Plot(injected_H1, start_index, Pos, detections, ensemble=4):
    if ensemble == 3:
        Pos = Pos
    elif ensemble == 2:
        Pos = np.intersect1d(detections[0], detections[1])
    else:
        Pos = detections[3]
        
    y    = np.zeros(injected_H1.shape[0])
    for index in range(start_index.shape[0]):
        y[start_index[index]*4096:(start_index[index]+1)*4096] = 1

    pred = np.zeros(injected_H1.shape[0])
    for p in range(Pos.shape[0]):
        pred[Pos[p]*4096:(Pos[p]+1)*4096] = 1

    larger_arr = np.union1d(Pos,start_index)
    
    fig, axs = plt.subplots(4, 3, figsize=(15, 15)) # Adjust size as needed
    axs = axs.ravel()  # Flatten the array of axes

    for i in range(larger_arr.shape[0]):
        start_time = (larger_arr[i]//4*4)*4096
        end_time = (larger_arr[i]//4*4+4)*4096
        x = np.linspace(larger_arr[i]//4*4, larger_arr[i]//4*4+4, end_time - start_time)

        axs[i].plot(x, injected_H1[start_time:end_time], label='Strain')
        axs[i].plot(x, y[start_time:end_time], label='Ground Truth')
        axs[i].plot(x, pred[start_time:end_time], label='Prediction')

        axs[i].set_xticks(np.arange(larger_arr[i]//4*4, larger_arr[i]//4*4+4+1, 1))
        axs[i].set_xlabel(f'{larger_arr[i]//4*4} - {larger_arr[i]//4*4+4} s')
        axs[i].legend()
        axs[i].set_title(f'True Positive/Detection location: {larger_arr[i]}s')

    # Remove extra subplots
    for i in range(larger_arr.shape[0], 4*3):
        fig.delaxes(axs[i])

    plt.tight_layout()
    plt.show()

def sub_module(inp_shape=(4096, 1)):
    
    # convolutional operation parameters
    n_filters = 32 
    filter_width = 2
    dilation_rates = [2**i for i in range(11)] * 3 

    # define an input history series and pass it through a stack of dilated causal convolution blocks. 
    Input_seq = Input(shape=inp_shape, dtype='float32')
    x = Input_seq

    skips = []
    for dilation_rate in dilation_rates:

        # preprocessing - equivalent to time-distributed dense
        x = Conv1D(16, 1, padding='same', activation='relu')(x) 

        # filter convolution
        x_f = Conv1D(filters=n_filters,
                     kernel_size=filter_width, 
                     padding='same',
                     dilation_rate=dilation_rate)(x)

        # gating convolution
        x_g = Conv1D(filters=n_filters,
                     kernel_size=filter_width, 
                     padding='same',
                     dilation_rate=dilation_rate)(x)

        # multiply filter and gating branches
        z = Multiply()([Activation('tanh')(x_f),
                        Activation('sigmoid')(x_g)])

        # postprocessing - equivalent to time-distributed dense
        z = Conv1D(16, 1, padding='same', activation='relu')(z)

        # residual connection
        x = Add()([x, z])    

        # collect skip connections
        skips.append(z)

    # add all skip connection outputs 
    out = Activation('relu')(Add()(skips))
    
    sub_module = Model(Input_seq, out)
    
    return sub_module


def GNN(target_node, neighbor_1, neighbor_2, Neighbor_Aggregation, Update_TargetNode, dim):
    
    neighbor_1  = Neighbor_Aggregation(neighbor_1); neighbor_1 = Reshape(target_shape=(4096, dim, 1))(neighbor_1)
    neighbor_2  = Neighbor_Aggregation(neighbor_2); neighbor_2 = Reshape(target_shape=(4096, dim, 1))(neighbor_2)
    
    neighbors   = tf.keras.layers.Concatenate(axis=-1)([neighbor_1, neighbor_2])    
    neighbors   = tf.reduce_max(neighbors, axis=-1)
    

    out_node = tf.keras.layers.Concatenate(axis=-1)([target_node, neighbors])
    out_node = Update_TargetNode(out_node)
    
#     out_node = Lambda(lambda x: K.l2_normalize(x,axis=-2))(out_node)
    
    return out_node


def full_module(inp_shape=(4096, 3)):
    
    # define an input history series and pass it through a stack of dilated causal convolution blocks. 
    Input_seq = Input(shape=inp_shape, dtype='float32')
    
    x_A = Lambda(lambda y: y[:,:,0])(Input_seq); x_A = Reshape(target_shape=(4096,1))(x_A)
    x_B = Lambda(lambda y: y[:,:,1])(Input_seq); x_B = Reshape(target_shape=(4096,1))(x_B)
    x_C = Lambda(lambda y: y[:,:,2])(Input_seq); x_C = Reshape(target_shape=(4096,1))(x_C)
    
    sub_mod_A = sub_module(inp_shape=(4096, 1))
    sub_mod_B = sub_module(inp_shape=(4096, 1))
    sub_mod_C = sub_module(inp_shape=(4096, 1))
    
    x_A   = sub_mod_A(x_A)
    x_B   = sub_mod_B(x_B)
    x_C   = sub_mod_C(x_C)
    
    dim = 16
    node_dim = 64
    
    # Layer Initialization
    Neighbor_Aggregation = Conv1D(dim, 1, activation = 'relu', name = 'Neighbor_Aggregation')
    Update_TargetNode    = Conv1D(node_dim, 1, activation = 'relu', name = 'Update_TargetNode')

    # Update Node
    Updated_x_A   = GNN(x_A, x_B, x_C, Neighbor_Aggregation, Update_TargetNode, dim);  Updated_x_A = Reshape(target_shape=(4096, node_dim, 1))( Updated_x_A)
    Updated_x_B   = GNN(x_B, x_A, x_C, Neighbor_Aggregation, Update_TargetNode, dim);  Updated_x_B = Reshape(target_shape=(4096, node_dim, 1))( Updated_x_B)
    Updated_x_C   = GNN(x_C, x_A, x_B, Neighbor_Aggregation, Update_TargetNode, dim);  Updated_x_C = Reshape(target_shape=(4096, node_dim, 1))( Updated_x_C)

    
    # Graph-level prediction
    out   = tf.keras.layers.Concatenate(axis=-1, name = 'Updated_Nodes_Concat')([Updated_x_A, Updated_x_B, Updated_x_C])
    out   = tf.reduce_max(out, axis=-1)    
    
    out   = Conv1D(1, 1)(out)
    out   = Activation('sigmoid')(out)
    out   = Conv1D(1, 4096)(out)
    out   = Activation('relu')(out)

    
    model = Model(Input_seq, out)
    
    return model

