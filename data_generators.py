# Standard python numerical analysis imports:
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
import tensorflow as tf
import h5py 
import json
import glob
import pickle
import matplotlib.mlab as mlab
from noise_snr_schedule import *



def whiten(strain, interp_psd, dt):
    Nt = len(strain)
    freqs = np.fft.rfftfreq(Nt, dt)

    # whitening: transform to freq domain, divide by asd, then transform back, 
    # taking care to get normalization right.
    hf = np.fft.rfft(strain)
    norm = 1./np.sqrt(1./(dt*2))
    white_hf = hf / np.sqrt(interp_psd(freqs)) * norm
    white_ht = np.fft.irfft(white_hf, n=Nt)
    return white_ht



def whiten_signal(strain_L1, strain_H1, strain_V1, dt, psd_L1, psd_H1, psd_V1 ):
    
    strain_whiten_L = whiten(strain_L1, psd_L1, dt)
    strain_whiten_H = whiten(strain_H1, psd_H1, dt)
    strain_whiten_V = whiten(strain_V1, psd_V1, dt)
    

    strain_whiten_L /= np.amax(np.absolute(strain_whiten_L))
    strain_whiten_H /= np.amax(np.absolute(strain_whiten_H))
    strain_whiten_V /= np.amax(np.absolute(strain_whiten_V))
    
    
    return strain_whiten_L, strain_whiten_H, strain_whiten_V



def get_whitened_ligo_noise_chunk(strain, noise_strain_fn, gaussian = 0):
    
    f = h5py.File(noise_strain_fn, 'r')
    strain_L1 = f['strain_L1']
    strain_H1 = f['strain_H1']
    strain_V1 = f['strain_H1']
    if gaussian:
        strain_V1 = f['strain_V1']
        
    # 4096
    # Get a random chunk of noise strain
    starting_index = np.random.randint(0, len(strain_H1)-len(strain))
    
    ligo_noise_L = np.zeros(len(strain))
    ligo_noise_H = np.zeros(len(strain))
    ligo_noise_V = np.zeros(len(strain))
    
    ligo_noise_L[:] = strain_L1[starting_index:starting_index+len(strain)]
    ligo_noise_H[:] = strain_H1[starting_index:starting_index+len(strain)]
    ligo_noise_V[:] = strain_V1[starting_index:starting_index+len(strain)]
    
    f.close()
    
    return ligo_noise_L, ligo_noise_H, ligo_noise_V
    

    
def mix_signal_and_noise(strain_whiten_L, strain_whiten_H, strain_whiten_V, ligo_noise_L, ligo_noise_H, ligo_noise_V, noise_range):
    # Mix with Noise
    
    target_std = np.random.uniform(noise_range[0], noise_range[1])
    

    ligo_noise_whiten_std_L = np.std(ligo_noise_L)
    ligo_noise_whiten_std_H = np.std(ligo_noise_H)
    ligo_noise_whiten_std_V = np.std(ligo_noise_V)
    

    ligo_noise_whiten_L = target_std*(ligo_noise_L/ligo_noise_whiten_std_L)
    ligo_noise_whiten_H = target_std*(ligo_noise_H/ligo_noise_whiten_std_H)
    ligo_noise_whiten_V = target_std*(ligo_noise_V/ligo_noise_whiten_std_V)
    

    mixed_L = strain_whiten_L + ligo_noise_whiten_L
    mixed_H = strain_whiten_H + ligo_noise_whiten_H
    mixed_V = strain_whiten_V + ligo_noise_whiten_V

    mixed_std_L = np.std(mixed_L)
    mixed_std_H = np.std(mixed_H)
    mixed_std_V = np.std(mixed_V)
    
    mixed_L = mixed_L / mixed_std_L
    mixed_H = mixed_H / mixed_std_H
    mixed_V = mixed_V / mixed_std_V

    return mixed_L, mixed_H, mixed_V



def single_shift(strain, merger, merger_shift, truncation):
    
    #     target = np.zeros(4096)
    target = np.zeros(8192)
    target[max(truncation, merger-2048):merger+1] = 1
    
    
    if merger_shift <= merger:
        strain = strain[merger-merger_shift:merger-merger_shift+4096]
        target = target[merger-merger_shift:merger-merger_shift+4096]
        
    else:
        tmp        = strain[:4096-(merger_shift-merger)]
        tmp_target = target[:4096-(merger_shift-merger)]

        strain = np.zeros(4096)
        target = np.zeros(4096)
        
        strain[merger_shift-merger:] = tmp[:]
        target[merger_shift-merger:] = tmp_target[:]

        
    return strain, target




class wfGenerator(tf.keras.utils.Sequence):
    def __init__(self, noise_dir, data_dir, batch_size=32, dim=(4096,), n_channels=2,
                 shuffle=True, spin=1, noise_prob=0.6, noise_range=None, hvd_rank=0, hvd_size=1):
        'Initialization'
        self.data_dir = data_dir
        self.noise_dir = noise_dir
        
        # Load Waveform Filename
        if spin:
            self.wf_file = self.data_dir + 'train_1.hdf'
        else:
            self.wf_file = self.data_dir + 'test_1.hdf'

        
        # Load Noise Filenames
        self.psd_L1_files = self.noise_dir + 'psd_L.pkl'
        self.psd_H1_files = self.noise_dir + 'psd_H.pkl'
        self.psd_V1_files = self.noise_dir + 'psd_V.pkl'
        self.noise_files  = sorted(glob.glob(self.noise_dir + 'gaussian_4096_*'))
            
        
        # Open waveforms file
        self.f = h5py.File(self.wf_file, 'r')
        self.keys = list(self.f.keys())
        # L1, H1, V1 waveforms (i, 8192)
        self.L1_wset = self.f[self.keys[0]+"/L1_wave"]
        self.H1_wset = self.f[self.keys[0]+"/H1_wave"]
        self.V1_wset = self.f[self.keys[0]+"/V1_wave"]
        
        # Other meta info.
        self.fs = 4096
        self.dt = 1/self.fs
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.noise_prob = noise_prob
        self.noise_range = noise_range
        self.hvd_rank = hvd_rank
        self.hvd_size = hvd_size
        self.epoch = 0
        self.on_epoch_end()

    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.local_indices) / self.batch_size))

    
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        pos = self.local_indices[index*self.batch_size:(index+1)*self.batch_size]
        if self.shuffle == True: #Sorting is necessary to read h5 files
            pos = np.sort(pos)

        # Generate data
        X, y = self.__data_generation(pos)

        return X, y

    
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.global_indices = np.arange(len(self.L1_wset))
        if self.shuffle == True:
            np.random.seed(self.epoch); np.random.shuffle(self.global_indices)
        
        chunk_size = len(self.global_indices) // self.hvd_size
        self.local_indices = self.global_indices[self.hvd_rank*chunk_size: (self.hvd_rank + 1)*chunk_size]
        self.epoch += 1

    
    
    
    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'
        
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype = np.float32)
        y = np.empty((self.batch_size, *self.dim, 1), dtype=int)
        
        
        # Get the strains: [indexes, 4096]
        strains_L1 = self.L1_wset[indexes, 2048:-2048]
        strains_H1 = self.H1_wset[indexes, 2048:-2048]
        strains_V1 = self.V1_wset[indexes, 2048:-2048]
        #print(indexes)
        
        # Whiten and mix each strain
        for i in range(strains_L1.shape[0]):

            # With 40% probability we feed signal
            if np.random.random_sample() > self.noise_prob:
                
                # Load waveform (4096 wave + 4096 zero)
                #Append zeros to L1 wave
                strain_L1 = np.zeros(2*strains_L1.shape[1])
                strain_L1[:strains_L1.shape[1]] = strains_L1[i]
                #Append zeros to L1 wave
                strain_H1 = np.zeros(2*strains_H1.shape[1])
                strain_H1[:strains_H1.shape[1]] = strains_H1[i]
                #Append zeros to L1 wave
                strain_V1 = np.zeros(2*strains_V1.shape[1])
                strain_V1[:strains_V1.shape[1]] = strains_V1[i]        
                #Set wave merger location
                merger_L1 = np.argmax(np.absolute(strain_L1[:]))
                merger_H1 = np.argmax(np.absolute(strain_H1[:]))
                merger_V1 = np.argmax(np.absolute(strain_V1[:]))
#                 #
#                 strain_L1[:max(0, merger_L1-2048)] = 0
#                 strain_H1[:max(0, merger_H1-2048)] = 0
#                 strain_V1[:max(0, merger_V1-2048)] = 0

                # Load PSDs and whitened noise strains                
                psd_L1 = pickle.load(open(self.psd_L1_files, 'rb'), encoding="bytes")
                psd_H1 = pickle.load(open(self.psd_H1_files, 'rb'), encoding="bytes")
                psd_V1 = pickle.load(open(self.psd_V1_files, 'rb'), encoding="bytes")
                whitened_noise_strain_fn = self.noise_files

                    
                ligo_noise_L, ligo_noise_H, ligo_noise_V  = get_whitened_ligo_noise_chunk(strain_L1[:4096], whitened_noise_strain_fn,
                                                                                          gaussian = self.gaussian)


                # Whiten the signal
                strain_whiten_L, strain_whiten_H, strain_whiten_V = whiten_signal(strain_L1, strain_H1, strain_V1, self.dt, psd_L1, psd_H1, psd_V1)
    
                truncation =  150
                strain_whiten_L[:truncation]  = 0
                strain_whiten_L[-truncation:] = 0
                
                strain_whiten_H[:truncation]  = 0
                strain_whiten_H[-truncation:] = 0
                
                strain_whiten_V[:truncation]  = 0
                strain_whiten_V[-truncation:] = 0
                
                
                # Randomly shift the strain
                merger_shift = np.random.randint(4096//2, 4096)
                strain_whiten_L, target_L = single_shift(strain_whiten_L, merger_L1, merger_shift, truncation)
                strain_whiten_H, target_H = single_shift(strain_whiten_H, merger_H1, merger_H1+(merger_shift-merger_L1), truncation)
                strain_whiten_V, target_V = single_shift(strain_whiten_V, merger_V1, merger_V1+(merger_shift-merger_L1), truncation)
                # target = ( (target_L + target_H + target_V)> 0 )*1.0

                # Mix signal and noise
                if not self.noise_range:
                    noise_range = low_max_snr(self.epoch, noise_range_map)
                else:
                    noise_range = self.noise_range
                mixed_L, mixed_H, mixed_V = mix_signal_and_noise(strain_whiten_L, strain_whiten_H, strain_whiten_V, \
                                                        ligo_noise_L,    ligo_noise_H,    ligo_noise_V, noise_range)
                

                X[i,:,0] = mixed_L
                X[i,:,1] = mixed_H
                X[i,:,2] = mixed_V
                
                y[i,:,0] = target_H
                # [i,:,0] = target
                
                
            # With 60% probability we feed just noise
            else:
                strain = np.zeros(strains_L1.shape[1])
                
                file_idx = np.random.randint(0, 3)
                whitened_noise_strain_fn = self.noise_files[file_idx]
                    
                ligo_noise_L, ligo_noise_H, ligo_noise_V  = get_whitened_ligo_noise_chunk(strain[:4096], whitened_noise_strain_fn,
                                                                                          gaussian = self.gaussian)
                
                X[i,:,0] = (ligo_noise_L/np.std(ligo_noise_L))
                X[i,:,1] = (ligo_noise_H/np.std(ligo_noise_H))
                X[i,:,2] = (ligo_noise_V/np.std(ligo_noise_V))
                y[i,:,0] = np.zeros(*self.dim)


        
        
        # Last shuffle so that mass ratios aren't fed in ascending order
        if self.shuffle == True:
            assert len(X) == len(y)
            p = np.random.permutation(len(X))
            X, y = X[p], y[p]
        
        return X, y