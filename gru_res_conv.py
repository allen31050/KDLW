"""
Things you can do:
    * Implement a model
    * Evaluate your models
    * Extract and try different features from raw data (WARNING: DIFFICULT)

Competition:
    https://www.kaggle.com/c/kkstream-deep-learning-workshop

Raw dataset:
    https://drive.google.com/drive/u/1/folders/1H6h9EKKFloRwJ5c92-C4YHJQaPCwqG20

Preprocessed dataset:
    https://drive.google.com/file/d/1k3JCadCKJAXcuems09Ffpac4ePJzx1KY/view?usp=sharing
"""

INP_SIZE = 896
OPT_SIZE = 28
PATIENCE = 5
BATCH_SIZE = 512 # 32 optimal but slow
EPOCHS = 40
VAL_SPLIT = 0.1 # 0.1 > 0.2
CKPT_PATH = "weights.best.hdf5"

import os
import re
import csv
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from keras.models import *
from keras.layers.core import *
from keras.layers import *
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau
from subprocess import check_output
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import Imputer

def get_model():
    inp = Input(shape = (INP_SIZE, 1))
    # features = Dense(64, activation = "sigmoid")(inp) # Naive Feature Net
    x = SpatialDropout1D(0.1)(inp)

    rnn_out = Bidirectional(GRU(128, return_sequences = True, dropout = 0.1, recurrent_dropout = 0.1))(x) 
    
    conv_out_1 = Conv1D(64, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform")(rnn_out)
    conv_out_2 = Conv1D(64, kernel_size = 3, padding = "same", kernel_initializer = "glorot_uniform")(conv_out_1)
    conv_out_sum = Add()([conv_out_1, conv_out_2])

    avg_pool = GlobalAveragePooling1D()(conv_out_sum)
    max_pool = GlobalMaxPooling1D()(conv_out_sum)
    pooled = concatenate([avg_pool, max_pool])

    x = Dropout(0.1)(pooled)
    x = Dense(OPT_SIZE, activation = "sigmoid")(x)

    model = Model(inputs = inp, outputs = x)
    model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])

    return model

class RocAucEvaluation(Callback):
    def __init__(self, filepath = CKPT_PATH, validation_data=(), interval=10, max_epoch = 20):
        super(Callback, self).__init__()

        self.interval = interval
        self.filepath = filepath
        self.stopped_epoch = max_epoch
        # self.best = 0
        self.X_val, self.y_val = validation_data
        self.y_pred = np.zeros(self.y_val.shape)

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, batch_size=512, verbose=0)
            
            # Important lines
            y_pred = Imputer().fit_transform(y_pred)
            current = roc_auc_score(self.y_val, y_pred)
            logs['roc_auc_val'] = current

            print(" - AUC - score: {:.5f}".format(current))
            
            global globalBest
            if current > globalBest: # self.best: #save model
            # if current > self.best: #save model
                # self.best = current
                globalBest = current
                self.y_pred = y_pred
                self.stopped_epoch = epoch + 1
                self.model.save(self.filepath, overwrite = True)
                print("AUC score improved, model saved\n")
            else:
                print("AUC score not improved\n")

def getCallbacks(valX, valY):
    RocAucVal = RocAucEvaluation(validation_data = (valX, valY), interval = 1)
    # earlyStop = EarlyStopping(monitor = 'roc_auc_val', patience = PATIENCE, mode = 'max', verbose = 1)
    # return [RocAucVal, earlyStop]
    return [RocAucVal]

def write_result(name, predictions):
    """
    """
    if predictions is None:
        raise Exception('need predictions')

    predictions = predictions.flatten()

    if not os.path.exists('./results/'):
        os.makedirs('./results/')

    path = os.path.join('./results/', name)

    with open(path, 'wt', encoding='utf-8', newline='') as csv_target_file:
        target_writer = csv.writer(csv_target_file, lineterminator='\n')

        header = [
            'user_id',
            'time_slot_0', 'time_slot_1', 'time_slot_2', 'time_slot_3',
            'time_slot_4', 'time_slot_5', 'time_slot_6', 'time_slot_7',
            'time_slot_8', 'time_slot_9', 'time_slot_10', 'time_slot_11',
            'time_slot_12', 'time_slot_13', 'time_slot_14', 'time_slot_15',
            'time_slot_16', 'time_slot_17', 'time_slot_18', 'time_slot_19',
            'time_slot_20', 'time_slot_21', 'time_slot_22', 'time_slot_23',
            'time_slot_24', 'time_slot_25', 'time_slot_26', 'time_slot_27',
        ]

        target_writer.writerow(header)

        for i in range(0, len(predictions), 28):
            # NOTE: 57129 is the offset of user ids
            userid = [57159 + i // 28]
            labels = predictions[i:i+28].tolist()

            target_writer.writerow(userid + labels)


# NOTE: load the data from the npz
dataset = np.load('./input/v0_eigens.npz')

# NOTE: calculate th size of training set and validation set
#       all pre-processed features are inside 'train_eigens'
train_data_size = dataset['train_eigens'].shape[0]
valid_data_size = int(train_data_size * VAL_SPLIT)
train_data_size = train_data_size - valid_data_size

# NOTE: split dataset
np.random.shuffle(dataset['train_eigens'])
train_data = dataset['train_eigens'][:train_data_size]
valid_data = dataset['train_eigens'][train_data_size:]

# NOTE: a 896d feature vector for each user, the 28d vector in the end are
#       labels
#       896 = 32 (weeks) x 7 (days a week) x 4 (segments a day)
train_eigens = train_data[:, :-28, np.newaxis].reshape(-1, 896, 1)
train_labels = train_data[:, -28:]

valid_eigens = valid_data[:, :-28, np.newaxis].reshape(-1, 896, 1)
valid_labels = valid_data[:, -28:]

# NOTE: read features of test set
test_eigens = dataset['issue_eigens'][:, :-28, np.newaxis].reshape(-1, 896, 1)

# NOTE: check the shape of the prepared dataset
print('train_eigens.shape = {}'.format(train_eigens.shape))
print('train_labels.shape = {}'.format(train_labels.shape))
print('valid_eigens.shape = {}'.format(valid_eigens.shape))
print('valid_labels.shape = {}'.format(valid_labels.shape))

model = get_model()
if os.path.isfile(CKPT_PATH):
    print("Loading last model...")
    model.load_weights(CKPT_PATH)

globalBest = 0
countEarlyStop = 0
for e in range(EPOCHS):
    lastgloBest = globalBest
    model.fit(train_eigens, train_labels, batch_size = BATCH_SIZE, epochs = 1, validation_data = (valid_eigens, valid_labels), callbacks = getCallbacks(valid_eigens, valid_labels), verbose = 2) # batch_size = min(1024, BATCH_SIZE * (2 ** e))
    if (globalBest == lastgloBest):
        countEarlyStop += 1
    else:
        countEarlyStop = 0
    if countEarlyStop >= PATIENCE:
        print("Early stopping...")
        break

# NOTE: predict and save
model.load_weights(CKPT_PATH)
test_labels = model.predict(test_eigens, batch_size = 512, verbose = 2)

write_result('sub.csv', test_labels)

