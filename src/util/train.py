from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
from python_speech_features import ssc
import scipy.io.wavfile as wav
import pandas as pd
import numpy as np

models = ['svm']
features = {'mfcc': mfcc, 'd_mfcc': d_mfcc, 'fbank': logfbank, 'ssc': ssc}

# TODO: modularize d_mfcc
def d_mfcc():
    """
    check notebook/svm_test.ipynb
    """
    return None

# TODO: train the model and save in the database
def train(trainset, model='svm', feature='mfcc'):
    """
    :param: model: model to train with
    :param trainset: training data filename
    :param nclass: number of classes
    :return:
    """
    if model not in models:
        raise('model not supported')
    if feature not in features:
        raise('feature not supported')

    # number of classes
    nclass = len(trainset)
    rate = []*nclass
    signal = []*nclass
    for i in range(nclass):
        # wav file to signal
        (rate[i], signal[i]) = wav.read(trainset[i])
    # calculate feature
    # concatenate features of every trainset
    # generate a model
    # fit(train) the model
    # save the model in /tmp using joblib
    # store the model in database


# test the function!
if __name__ == '__main__':
    train_data = ['../data/test_sound/sound1.wav',
        '../data/test_sound/sound2.wav']
    
    train(train_data)


