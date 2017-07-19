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


CHUNK_SIZE = 8192
AUDIO_FORMAT = pyaudio.paInt16
SAMPLE_RATE = 16000
BUFFER_HOURS = 12
AUDIO_SERVER_ADDRESS = ('localhost', 6000)
WAVE_FILENAME = 'test/test2.wav'
dumps = {'svm': {'mfcc': '../models/svm_mfcc.joblib.pkl'}}
feature_func = {'mfcc': mfcc}

def process_audio(shared_audio, shared_time, shared_pos, lock):
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
    stream = p.open(format=AUDIO_FORMAT, channels=1, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK_SIZE)
    wf = wave.open(WAVE_FILENAME, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(AUDIO_FORMAT))
    wf.setframerate(SAMPLE_RATE)

    try:
        while True:
            # grab audio and timestamp
            audio = np.fromstring(stream.read(CHUNK_SIZE), np.int16)
            current_time = time.time()

            # acquire lock
            lock.acquire()

            # record current time
            shared_time[shared_pos.value] = current_time

            # record the maximum volume in this time slice
            shared_audio[shared_pos.value] = np.abs(audio).max()
            wf.writeframes(b''.join(audio))

            # increment counter
            shared_pos.value = (shared_pos.value + 1) % len(shared_time)

            # release lock
            lock.release()
    except KeyboardInterrupt:
        stream.stop_stream()
        stream.close()
        p.terminate()

    # after exiting the loop
    stream.stop_stream()
    stream.close()
    p.terminate()

def process_requests(shared_audio, shared_time, shared_pos, lock):
    """
    Handle requests from the web server. First get the latest data, and
     then analyse it to find the current noise state
    :param shared_audio:
    :param shared_time:
    :param shared_pos:
    :param lock:
    :return:
    """

    listener = Listener(AUDIO_SERVER_ADDRESS)
    while True:
        conn = listener.accept()

        # get some parameters from the client
        parameters = conn.recv()

        # acquire lock
        lock.acquire()

        # convert to numpy arrays and get a copy of the data
        time_stamps = np.frombuffer(shared_time, np.float64).copy()
        audio_signal = np.frombuffer(shared_audio, np.int16).astype(np.float32)
        current_pos = shared_pos.value

        # release lock
        lock.release()

        # TODO: preprocess and predict
        """
        1. resample to x!
        2. pre-process(normalize ... ) -> later
        3. retrieve model object from db (for now from a model file)
        4. predict
        5. logic for calculating 
        results = {'audio_plot': audio_plot,
                   'y_pred': y_pred
                   }
        """
        # existing code in little sleeper
        normalized = normalize(audio_signal, time_stamps, current_pos, parameters, SAMPLE_RATE, CHUNK_SIZE)
        audio_plot = normalized[0]
        audio_signal = normalized[1]
        time_stamps = normalized[2]

        # existing code in little sleeper
        results = process_signal(audio_plot, audio_signal, parameters, 
                time_stamps)

        conn.send(results)
        conn.close()

def predict_audio(sig, feat='mfcc', model='svm'):
    """
    calculate feature from signal
    retrieve model from dumps dictionary
    predict
    """
    feature = featfunc[feat](sig)
    dump = dumps[model][feat]
    clf = joblib.loads(dump)
    y_pred = clf.predict(feature)
    return y_pred

# TODO: make a logic to detect uav sound from prediction
def detect_uav(y_pred):
    results = None
    return results

def init_server():
    # figure out how big the buffer needs to be to contain BUFFER_HOURS of audio
    buffer_len = int(BUFFER_HOURS * 60 * 60 * (SAMPLE_RATE / float(CHUNK_SIZE)))

    # create shared memory
    lock = mp.Lock()
    shared_audio = mp.Array(ctypes.c_short, buffer_len, lock=False)
    shared_time = mp.Array(ctypes.c_double, buffer_len, lock=False)
    shared_pos = mp.Value('i', 0, lock=False)

    # start 2 processes:
    # 1. a process to continuously monitor the audio feed
    # 2. a process to handle requests for the latest audio data
    p1 = mp.Process(target=process_audio, args=(shared_audio, shared_time, shared_pos, lock))
    p2 = mp.Process(target=process_requests, args=(shared_audio, shared_time, shared_pos, lock))
    p1.start()
    p2.start()


if __name__ == '__main__':
    init_server()
