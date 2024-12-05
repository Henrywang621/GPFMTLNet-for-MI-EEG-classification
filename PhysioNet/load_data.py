import numpy as np
import random
from mne import pick_types
from mne.io import read_raw_edf
from sklearn.preprocessing import scale
import numpy as np
from sklearn.preprocessing import scale, OneHotEncoder

def create_trial(start_index, run_length, trial_duration):
    '''
    input:
    start_index - the onset of the current run
    run_length - the number of samples in the run
    trial_duration - the constructed trial in seconds 
    
    output: a list of tuple ranges
    '''

    sampling_rate = 160
    trial_length = int(trial_duration * sampling_rate)
    pad = (trial_length - run_length) // 2
    reminder = (trial_length - run_length) % 2
    end_index = start_index + run_length
    windows = []
    windows += [round(start_index - pad - reminder), round(end_index + pad)]
    return windows

def getOneHotLabels(y):
    oh = OneHotEncoder()
    out = oh.fit_transform(y.reshape(-1, 1)).toarray()
    return out

def construct_X(event, data):
    '''Segment a run'''
    
    sampling_rate = 160
    trial_duration = 6
    start_pos = int(event[0] * sampling_rate)
    run_length = int(event[1] * sampling_rate)
    windows = create_trial(start_pos, run_length, trial_duration)
    # downsample each segement to 128Hz
    x = data[:, windows[0]: windows[1]]
    # x = mne.filter.resample(data[:, windows[0]: windows[1]], down=0.8, npad='auto')
    return x

def read_edffile(subject_id, current_run):
    base_path = '/home/henrywang/PythonPractice/testCP_RCNN/datasets/BCI2000/S{0:03}/S{1:03}R{2:02}.edf'
    path = base_path.format(subject_id, subject_id, current_run)
    raw = read_raw_edf(path, preload=True, verbose=False)
    onset = raw.annotations.onset
    duration = raw.annotations.duration
    description = raw.annotations.description
    # events[Oneset of the event in seconds, Duration of the event in seconds, Description]
    events = [[onset[i], duration[i], description[i]] for i in range(len(raw.annotations))]
    picks = pick_types(raw.info, eeg=True)
    raw.filter(l_freq=4, h_freq=None, picks=picks)
    data = raw.get_data(picks=picks)
    
    return events, data


'''
Experimental groups of Binary classification

Group1: Left hand & Right hand -> T1-run_type2 (label:0) & T2-run_type2 (label:1);
'''

def load_data(subject_ids):
    
    trials = []
    labels = []
    runs_type = [4, 8, 12]


    for subject_id in subject_ids:
        for MI_run in runs_type:
            events, data = read_edffile(subject_id, MI_run)
            for event in events[:-1]:
                if event[2] == 'T0':
                    continue
                else:
                    x = construct_X(event, data)
                    y = [0] if event[2] == 'T1' else [1]
                    trials.append(x)
                    labels += y

    return np.array(trials), np.array(labels).reshape(-1, 1)

def norm_data(X):
    
    # Z-score Normalization
    shape = X.shape
    for i in range(shape[0]):
        X[i,:, :] = scale(X[i,:, :])
        if (i+1)%int(shape[0]//10) == 0:
            print('{:.0%} done'.format((i+1)/shape[0]))
    
    return X
    
def load_EEGdata4PhysioNet(subject_ids):
    X, y = load_data(subject_ids)
    X = np.hstack((X[:, 0:2, :], X))
    X = np.hstack((X, X[:, -2:, :]))

    # Z-score Normalization
    shape = X.shape
    for i in range(shape[0]):
        X[i,:, :] = scale(X[i,:, :]) 
        # if (i+1)%int(shape[0]//10) == 0:
        #     print('{:.0%} done'.format((i+1)/shape[0]))
    
    return np.swapaxes(X, 1, 2), getOneHotLabels(y)

def create_data4SSL(data, trial_length, overlap):
    '''
        data: its shape should be (num_shot, samples, n_chans)
        trial_length: the length of the whole trial.
        overlap: the overlap between two neighbouring trials.
        return: support data with the shape (num_shot, samples, n_chans) and labels with the same shape used for self-supervised learning
    '''

    if trial_length > data.shape[1]:
       raise ValueError("Please check the length of the trials you set.") 
    
    elif trial_length*2 - overlap > data.shape[1]:
       raise ValueError("Please check the length of the overlap you set.")
    
    elif overlap > trial_length:
       raise ValueError("overlap should not be greater than trial_length.")

    start_pos = trial_length-overlap-1
    end_pos = 2*trial_length-overlap-1

    return data[:,:trial_length,:], data[:,start_pos:end_pos,:]


def exp_dataDim(data, std=0.5):

    '''
    Increase the signals' dimension through adding noise according to the way usd in C-LSTM:
    data_noise = data + rand*stddev(data)/C_noise, where rand is the randomly generated number from uniform distribution
    [-0.5, 0.5]. C_noise is set to 3 here.
    '''

    rand1 = np.random.uniform(low=-0.5,high=0.5)
    rand2 = np.random.uniform(low=-0.5,high=0.5)
    X_noise1 = data+rand1*std/3
    X_noise2 = data+rand2*std/3
    insert_pos1 = list(range(1, 66, 3))
    insert_pos2 = list(range(2, 66, 3))

    for i in range(data.shape[2]):
        data = np.insert(data, insert_pos1[i], X_noise1[:,:,i],axis=2)
        data = np.insert(data, insert_pos2[i], X_noise2[:,:,i],axis=2)
        
    return data

def spilt_dataset(subject_ids, i):
    seeds = [42, 68, 188, 256, 1337]
    subject_ids = np.random.RandomState(seed=seeds[i]).permutation(subject_ids)
    ids_length = len(subject_ids)
    train_ids, val_ids, test_ids = np.split(subject_ids, [int(.7 * ids_length), int(.8 * ids_length)])
    return train_ids, val_ids, test_ids


def batch_generator4SS(classes, shot_num):

    while True:
        subj_id = random.sample(list(range(9)), 1)
        save_path = '/home/henrywang/testEEGModels/BCIIV2a/data/'
        data_X = np.load(save_path+'S{0}_X.npy'.format(subj_id[0]+1))

        # label:1 -> 'left_hand' & label:2 -> 'right_hand'
        data_y = np.load(save_path+'S{0}_y.npy'.format(subj_id[0]+1))

        # calculate STD of the data
        std = np.std(data_X)

        # obtain MI-EEG data with the specific classes
        c1, c2 = classes
        c1_index = np.argwhere(data_y==c1)
        c2_index = np.argwhere(data_y==c2)
        c12_index = np.sort(np.concatenate([np.squeeze(c1_index, axis=-1), np.squeeze(c2_index, axis=-1)], axis=0))

        data = data_X[c12_index]
        data = np.swapaxes(data, 1, 2)

        data = exp_dataDim(data, std=std)

        tr_data, labels = create_data4SSL(data, 960, 800)
        ids = random.sample(list(range(tr_data.shape[0])), shot_num)
        inp = tr_data[ids][:,:,:64]
        lbs = labels[ids][:,:,:64]
        yield inp, lbs



def batch_generator4S2(dataset_name, subjects_id, shot_num, training=True):

    while True:

        if dataset_name == 'PhysioNet':
            if training:
                subj_id = random.sample(subjects_id, 3)
                data_X, label_y = load_EEGdata4PhysioNet(subj_id)

            else:
                data_X, label_y = load_EEGdata4PhysioNet(subjects_id)

        else:
            raise ValueError("The dataset you choose is not included in this program!!!")
        
        ids1 = random.sample(list(range(data_X.shape[0])), shot_num)
        ids2 = random.sample(list(range(data_X.shape[0])), shot_num)
        Inp4IN = data_X[ids1, :, :64]
        Inp4PN = data_X[ids2, :, :64]
        lab4PN = label_y[ids2]

        yield ((Inp4IN, Inp4PN), lab4PN)