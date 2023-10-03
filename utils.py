from __future__ import print_function
import h5py
import numpy as np
import scipy.signal
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from time import time
import multiprocessing
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 
from tensorflow_addons.optimizers import LAMB

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
#     data = np.stack(( whitened_L1, whitened_H1, whitened_V1), axis=1)

    # Create datagenerators
    dg_0 = TimeseriesGenerator(data=data, targets=data, length=4096, stride=4096, start_index=0, batch_size=256)
    dg_5 = TimeseriesGenerator(data=data, targets=data, length=4096, stride=4096, start_index=2047, batch_size=256)
    
    # Make preds
    preds_0 = Model.predict(dg_0, verbose=1)
    preds_5 = Model.predict(dg_5, verbose=1)
    
    return preds_0.ravel(), preds_5.ravel()
#     return preds_5.ravel()


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
