import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.losses import CategoricalCrossentropy
import numpy as np
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, roc_auc_score
from load_data import *
from SSNetV2 import *
# from loss import *


def mse_segmentation_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


subject_ids = list(range(1, 110))
# Damaged recordings (#88, #89, #92, #100 and #104) need to be removed.
remove_ids = [88, 89, 92, 100, 104]

# remove subjects'data sampled at 128Hz
for id in remove_ids:
    subject_ids.remove(id)



epochs = 500
num_chans = 68
batch_size = 16
folds = range(5)
results = np.zeros((len(folds), 4))
# lr = 1e-5 
# adam = Adam(learning_rate = lr)
lossWeights = {"output1": 1, "output2": 2}

for i in folds:
    train_ids, val_ids, test_ids = spilt_dataset(subject_ids, i)
    X_train, y_train = load_EEGdata4PhysioNet(train_ids)
    X_val, y_val = load_EEGdata4PhysioNet(val_ids)
    # X_test, y_test = load_EEGdata4PhysioNet(val_ids)
    print(X_train.shape)
    X_train = np.expand_dims(X_train, axis=-1)
    X_val = np.expand_dims(X_val, axis=-1)
    # X_test = np.expand_dims(X_test, axis=-1)

    model = SSNetV2(trial_length = 960, nchans = num_chans)
    
    model.compile(loss={'output1': mse_segmentation_loss, 'output2': 'categorical_crossentropy'}, loss_weights=lossWeights, optimizer='adam', 
                  metrics = {'output1': 'MeanSquaredError', 'output2': 'accuracy'})
    earlystopping = EarlyStopping(monitor='val_loss', patience=100)
    weights_path = "Checkpoints/SSNetV2-{0}.hdf5".format(i)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, 
                    factor=0.5, mode='min', verbose=1, 
                    min_lr=1e-5)
    model_checkpoint = ModelCheckpoint(filepath = weights_path, monitor='val_output2_loss', 
                                       verbose=1, save_best_only=True)
    callback_list = [model_checkpoint, earlystopping, reduce_lr]
    fittedModel = model.fit([X_train, X_train], {'output1': X_train, 'output2': y_train}, epochs = epochs, batch_size = batch_size, validation_data = ([X_val, X_val], {'output1': X_val, 'output2': y_val}), verbose= 2, shuffle = True, callbacks=callback_list)







