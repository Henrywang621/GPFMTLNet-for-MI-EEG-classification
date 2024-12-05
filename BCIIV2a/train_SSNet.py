import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.losses import CategoricalCrossentropy
import numpy as np
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, roc_auc_score
from utils import *
import mne
from SSNetV2 import *
from tensorflow.keras.callbacks import ReduceLROnPlateau
import argparse
# from loss import *




def mse_segmentation_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


parser = argparse.ArgumentParser(description="Please input the number of epoches and the value of early stopping patience.")
parser.add_argument("--numepochs", type=int)
parser.add_argument("--Patience4ES", type=int)
args = parser.parse_args()
Epochs= args.numepochs
Patience = args.Patience4ES

save_path = '/home/henrywang/testEEGModels/BCIIV2a/data/'
# random_seeds = [72, 66, 18, 46, 11]
# random_seeds = [72, 66, 42, 46, 11]
# random_seeds = [42, 16, 20, 7, 95]
# random_seeds = [13, 15, 9, 6, 23]
random_seeds = [72, 66, 13, 46, 15]

# results = np.zeros((len(subjects), len(folds) + 1, 4))
nclasses = 4
nchans = 22
epochs = Epochs
batch_size = 64
lossWeights = {"output1": 1, "output2": 2}
results = np.zeros((6, 4))


for i in range(5):
    subjs_tr, subjs_va, subjs_te = split_datasets(seed=random_seeds[i])
    # data_X_tr = [mne.filter.resample(np.load(save_path+'S{0}_X.npy'.format(i)), down = 1.171875, npad='auto') for i in subjs_tr]
    data_X_tr = [mne.filter.resample(np.load(save_path+'S{0}_X.npy'.format(i)), up = 1.088, npad='auto') for i in subjs_tr]
    data_trX_c = np.vstack(data_X_tr)

    # Clear the GPU memory of the variable data_X_tr
    del data_X_tr

    # data_X_va = [mne.filter.resample(np.load(save_path+'S{0}_X.npy'.format(j)), down = 1.171875, npad='auto') for j in subjs_va]
    data_X_va = [mne.filter.resample(np.load(save_path+'S{0}_X.npy'.format(j)), up = 1.088, npad='auto') for j in subjs_va]
    data_vaX_c = np.vstack(data_X_va)
    
    # Clear the GPU memory of the variable data_X_tr
    del data_X_va   

    # data_X_te = [mne.filter.resample(np.load(save_path+'S{0}_X.npy'.format(i)), down = 1.171825, npad='auto') for i in subjs_te]
    data_X_te = [mne.filter.resample(np.load(save_path+'S{0}_X.npy'.format(i)), up = 1.088, npad='auto') for i in subjs_te]
    data_teX_c = np.vstack(data_X_te)

    # Clear the GPU memory of the variable data_X_tr
    del data_X_te

    data_y_tr = [onehot_labels(np.load(save_path+'S{0}_y.npy'.format(i)).reshape(-1, 1)) for i in subjs_tr]
    data_trY_c = np.vstack(data_y_tr)

    # Clear the GPU memory of the variable data_X_tr
    del data_y_tr

    data_y_te = [onehot_labels(np.load(save_path+'S{0}_y.npy'.format(i)).reshape(-1, 1)) for i in subjs_te]
    data_teY_c = np.vstack(data_y_te)

    # Clear the GPU memory of the variable data_X_tr
    del data_y_te

    data_y_va = [onehot_labels(np.load(save_path+'S{0}_y.npy'.format(i)).reshape(-1, 1)) for i in subjs_va]
    data_vaY_c = np.vstack(data_y_va)

    # Clear the GPU memory of the variable data_X_tr
    del data_y_va

    data_trX_c = std(data_trX_c)
    data_trX_c = np.swapaxes(data_trX_c, 1, 2)
    data_trX_c = np.expand_dims(data_trX_c, axis = -1)
    data_vaX_c = std(data_vaX_c)
    data_vaX_c = np.swapaxes(data_vaX_c, 1, 2)
    data_vaX_c = np.expand_dims(data_vaX_c, axis = -1)
    data_teX_c = std(data_teX_c)
    data_teX_c = np.swapaxes(data_teX_c, 1, 2)
    data_teX_c = np.expand_dims(data_teX_c, axis = -1)

    model = SSNetV2(trial_length = 1224, nchans = nchans)
    # initial_learning_rate = 0.0001
    # optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)

    model.compile(loss={'output1': mse_segmentation_loss, 'output2': 'categorical_crossentropy'}, loss_weights=lossWeights, optimizer='adam', 
                    metrics = {'output1': 'MeanSquaredError', 'output2': 'accuracy'})
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, 
                        factor=0.5, mode='min', verbose=1, 
                        min_lr=1e-6)
    earlystopping = EarlyStopping(monitor='val_loss', patience=Patience)
    weights_path = "Checkpoints/SSNet-{0}.hdf5".format(i+1)
    model_checkpoint = ModelCheckpoint(filepath = weights_path, monitor='val_output2_loss', 
                                        verbose=1, save_best_only=True)
    callback_list = [model_checkpoint, earlystopping, reduce_lr]
    fittedModel = model.fit([data_trX_c, data_trX_c], {'output1': data_trX_c, 'output2': data_trY_c}, epochs = epochs, batch_size = batch_size, validation_data = ([data_vaX_c, data_vaX_c], {'output1': data_vaX_c, 'output2': data_vaY_c}), verbose= 2, shuffle = True, callbacks=callback_list)

