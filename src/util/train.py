from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
from python_speech_features import ssc
from sklearn.externals import joblib
from sklearn.metrics import f1_score
from sklearn import svm
import scipy.io.wavfile as wav
import pandas as pd
import numpy as np
import pickle

import os
import glob

from features import d_mfcc

models = ['svm']
features = {'mfcc': mfcc, 'd_mfcc': d_mfcc, 'fbank': logfbank, 'ssc': ssc}


def train_and_dump(zero_trainset, one_trainset,
    model='svm', feature='mfcc'):
    print('** train_and_dump start')

    ## troubleshooting
    if model not in models:
        raise('model not supported')
    if feature not in features:
        raise('feature not supported')

    zero_len = len(zero_trainset)
    one_len = len(one_trainset)

    zero_rates = []
    zero_signals = []
    one_rates = []
    one_signals = []

    ## load wav files
    for i in range(zero_len):
        (r, s) = wav.read(zero_trainset[i])
        zero_rates.append(r)
        zero_signals.append(s)
    for i in range(one_len):
        (r, s) = wav.read(one_trainset[i])
        one_rates.append(r)
        one_signals.append(s)

    ## calculate & concatenate features
    # calculated_features = np.empty(0)
    calculated_features = None
    actual_labels = np.empty(0)
    calculated_model = svm.SVC()    # -> 수정수정
    for i in range(zero_len):   #zero
        cf = features[feature](zero_signals[i], zero_rates[i], nfft=2048)
        if calculated_features is None:
            calculated_features = cf
        else:
            calculated_features = np.concatenate((calculated_features, cf), axis=0)
        actual_labels = np.concatenate((actual_labels, np.zeros(cf.shape[0])), axis=0) #suppose that cf[0] is size? of features
    for i in range(one_len):    # one
        cf = features[feature](one_signals[i], one_rates[i], nfft=2048)
        calculated_features = np.concatenate((calculated_features, cf), axis=0)
        actual_labels = np.concatenate((actual_labels, np.ones(cf.shape[0])), axis=0)

    calculated_model.fit(calculated_features, actual_labels)

    ## save the model
    model_filepath = ("./tmp/%s.%s.joblib.pkl" % (model, feature))
    joblib.dump(calculated_model, model_filepath, compress=9)
    print('** train_and_dump end')


def load_and_predict(model='svm', feature='mfcc',
    test_filepath=None):
    print('** load_and_predict start')

    ## troubleshooting
    if model not in models:
        raise('model not supported')
    if feature not in features:
        raise('feature not supported')
    if test_filepath is None:
        raise('test file\'s path is needed')
    
    model_filepath = ("./tmp/%s.%s.joblib.pkl" % (model, feature))
    ## if model doesn't exist
    if not os.path.isfile(model_filepath):
        raise('model doesn\'t exist')
    loaded_model = joblib.load(model_filepath)

    ## test file read
    (test_rate, test_sig) = wav.read(test_filepath)
    cf_test = features[feature](test_sig, test_rate, nfft=2048)
    y_pred = loaded_model.predict(cf_test)
    print("I think there's a %.2f%% probability of drone."
      % ((np.count_nonzero(y_pred)/y_pred.shape[0])*100))
    print('** load_and_predict end')


# test the function!
if __name__ == '__main__':
    indoor_data_path = '../../data/indoor_sound'
    outside_data_path = '../../data/outside_sound'
    drone_data_path = '../../data/drone_sound'
    test_wav_path = '../../data/indoor_sound/indoor170726-001.wav'

    # path check
    if not os.path.isdir(indoor_data_path) or not os.path.isdir(outside_data_path) or not os.path.isdir(drone_data_path):
        raise("data path doesn't exist")

    indoor_data = glob.glob(indoor_data_path+'/*.wav')
    outside_data = glob.glob(outside_data_path+'/*.wav')
    drone_data = glob.glob(drone_data_path+'/*.wav')
    zero_data = indoor_data + outside_data

    # if there is no train file
    if len(zero_data) is 0 or len(drone_data) is 0:
        raise("train data doesn't exist")
    

    ########### just for test!!
    # zero_data = ['../../data/indoor_sound/indoor170719-001.wav']
    # drone_data = ['../../data/drone_sound/drone170719-002.wav']


    train_and_dump(zero_trainset=zero_data,
        one_trainset=drone_data,
        model='svm', feature='mfcc')
    load_and_predict(model='svm', feature='mfcc',
        test_filepath=test_wav_path)