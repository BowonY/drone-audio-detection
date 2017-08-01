import pyaudio
import wave
import numpy as np
import time
import multiprocessing as mp
from multiprocessing.connection import Listener
import ctypes
from scipy import ndimage, interpolate
from datetime import datetime
from sklearn.externals import joblib
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
from python_speech_features import ssc
from util.features import d_mfcc
from util.uav_detection import process_signal
from util.uav_detection import format_time_difference
from util.uav_detection import normalize
from util.uav_detection import get_plots

BUFFER_HOURS = 12
AUDIO_SERVER_ADDRESS = ('localhost', 6000)
WAVE_FILENAME = 'test/test2.wav'
dumps = {'svm': {'mfcc': '../models/svm_mfcc_22050.joblib.pkl',
                 'fbank': '../models/svm_fbank_22050.joblib.pkl'}}
feature_func = {'mfcc': mfcc, 'fbank': logfbank}

#Original setting
#CHUNK_SIZE = 8192
#AUDIO_FORMAT = pyaudio.paInt16
#SAMPLE_RATE = 16000
CHUNK_SIZE = 16000
AUDIO_FORMAT = pyaudio.paInt16
SAMPLE_RATE = 22050

def process_audio():
    """
    Endless loop: Grab some audio from the mic and record the maximum
    :param shared_audio:
    :param shared_time:
    :param shared_pos:
    :param lock:
    :return:
    """
    # open default audio input stream
    p = pyaudio.PyAudio()
    stream = p.open(format=AUDIO_FORMAT, channels=1, rate=SAMPLE_RATE, 
            input=True, frames_per_buffer=CHUNK_SIZE)
    wf = wave.open(WAVE_FILENAME, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(AUDIO_FORMAT))
    wf.setframerate(SAMPLE_RATE)

    try:
        while True:
            # grab audio and timestamp
            audio = np.fromstring(stream.read(CHUNK_SIZE), np.int16)
            # save
            wf.writeframes(b''.join(audio))
            # predict
            y_pred = predict_audio(audio, feat='fbank')
            # calculate percentage of 'one's
            percent_detected = judge_uav(y_pred)

            print(percent_detected)

    except KeyboardInterrupt:
        stream.stop_stream()
        stream.close()
        p.terminate()

    # after exiting the loop
    stream.stop_stream()
    stream.close()
    p.terminate()


def judge_uav(y_pred):
    """
    return the percentage of 'one' from prediction array
    """
    if len(y_pred) <= 0:
        return 0
    mask = y_pred > 0
    one_cnt = len(y_pred[mask])
    percent = one_cnt / float(len(y_pred)) *100
    return percent

def predict_audio(sig, feat='mfcc', model='svm'):
    """
    calculate feature from signal
    retrieve model from dumps (pre-trained model dictionary)
    return predictions
    """
    feature = feature_func[feat](sig)
    print('feature',feature.shape)
    dump = dumps[model][feat]
    clf = joblib.load(dump)
    y_pred = clf.predict(feature)
    return y_pred

def init_server():
    print("audio server start")

    # start 2 processes:
    # 1. a process to continuously monitor the audio feed
    p1 = mp.Process(target=process_audio)
    p1.start()

if __name__ == '__main__':
    init_server()
