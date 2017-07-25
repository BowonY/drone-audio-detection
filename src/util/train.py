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

import os
import glob

models = ['svm']
features = {'mfcc': mfcc, 'd_mfcc': d_mfcc, 'fbank': logfbank, 'ssc': ssc}

# TODO: modularize d_mfcc
def d_mfcc():
    """
    check notebook/svm_test.ipynb
    """
    return None

# TODO: train the model and save in the database
# def train(trainset, model='svm', feature='mfcc'):
def train(zero_trainset, one_trainset, model='svm', feature='mfcc'):
    """
    :param zero_trainset: training data list (labeled zero)
    :param one_trainset: training data list (labeled one)
    :param model: model to train with
    :param feature:
    :return:
    """
    if model not in models:
        raise('model not supported')
    if feature not in features:
        raise('feature not supported')



    # test
    if model is not 'svm':
        raise('For now, svm supported')


    ## calculate feature
    ## generate a model
    ## fit(train) the model
    ## save the model in /tmp using joblib
    ## store the model in database
    zero_len = len(zero_trainset)
    one_len = len(one_trainset)

    zero_rates = []
    zero_signals = []
    one_rates = []
    one_signals = []

    for i in range(zero_len):
        (r, s) = wav.read(zero_trainset[i])
        zero_rates.append(r)
        zero_signals.append(s)
    for i in range(one_len):
        (r, s) = wav.read(one_trainset[i])
        one_rates.append(r)
        one_signals.append(s)


    ## concatenate features
    calculated_features = np.empty(0)
    actual_labels = np.empty(0)
    calculated_model = svm.SVC()    # -> 수정수정

    # zero
    for i in range(zero_len):
        cf = features[feature](zero_signals[i], zero_rates[i], nfft=2048)
        # calculated_features.append(cf)
        calculated_features = np.concatenate((calculated_features, cf), axis=0)
        actual_labels = np.concatenate((actual_labels, np.zeros(cf.shape[0])), axis=0) #suppose that cf[0] is size? of features
    # one
    for i in range(one_len):
        cf = features[feature](one_signals[i], one_rates[i], nfft=2048)
        # calculated_features.append(cf)
        calculated_features = np.concatenate((calculated_features, cf), axis=0)
        actual_labels = np.concatenate((actual_labels, np.ones(cf.shape[0])), axis=0)

    calculated_model.fit(calculated_features, actual_labels)


    ## predict
    y_pred = calculated_model.predict(actual_labels)

    print("f1 score: %s" % (f1_score(actual_labels, y_pred)))

    
    # number of classes
    # nclass = len(trainset)
    # rates = []
    # signals = []
    # for i in range(nclass):
    #     # wav file to signal
    #     (temp_rate, temp_signal) = wav.read(trainset[i])
    #     rates.append(temp_rate)
    #     signals.append(temp_signal)

    # # calculate feature
    # calc_feat_list = []
    # for i in range(nclass):
    #     calc_feat = features[feature](signals[i], rates[i], nfft=2048)
    #     if feature == 'd_mfcc':
    #         calc_feat = features[feature](calc_feat, 2)
    #     calc_feat_list.append(calc_feat)    # concatenate features of every trainset

    # y_train = np.zeros(None) #temp

    # # temp
    # if model != "svm":
    #     raise("other model is not ready. this is just for svm")
    # clf = svm.SVC()


    # # generate a model
    # # (just for svm)
    # filename = ('../../models/tmp/test_%s.joblib.pkl' % model)
    # # joblib.dump()

    # # fit(train) the model


    # # save the model in /tmp using joblib
    # # store the model in database



# test the function!
if __name__ == '__main__':
    # train_data = ['../data/test_sound/sound1.wav',
    #     '../data/test_sound/sound2.wav']

    indoor_data_path = '../data/indoor_data'
    drone_data_path = '../data/drone_data'

    # path check
    if not os.path.isdir(indoor_data_path) or os.path.isdir(drone_data_path):
        raise("data path doesn't exist")

    # indoor_data = [f for f in os.listdir(indoor_data_path) if os.path.isfile(os.path.join(indoor_data_path, f))]
    # drone_data = [f for f in os.listdir(drone_data_path) if os.path.isfile(os.path.join(drone_data_path, f))]
    indoor_data = glob.glob(indoor_data_path+'/*.wav')
    drone_data = glob.glob(drone_data_path+'/*.wav')

    if len(indoor_data) is 0 or len(drone_data) is 0:
        raise("train data doesn't exist")
    
    train(zero_trainset=indoor_data,
        one_trainset=drone_data)