#
# how to use
# 1. python merged_server.py
# 2. http://localhost:8090
#

import os
from datetime import datetime

import tornado.httpserver
import tornado.ioloop
import tornado.web
import tornado.websocket
import tornado.gen

import pyaudio
import wave
import numpy as np
import time
import ctypes
import multiprocessing as mp
from multiprocessing.connection import Listener
from multiprocessing.connection import Client
from scipy import ndimage, interpolate
from sklearn.externals import joblib
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
from python_speech_features import ssc

AUDIO_SERVER_ADDRESS = ('localhost', 6000)
WEB_SERVER_ADDRESS = ('0.0.0.0', 8090)

UPPER_LIMIT = 25000
NOISE_THRESHOLD = 0.25

CHUNK_SIZE = 8192
AUDIO_FORMAT = pyaudio.paInt16
SAMPLE_RATE = 16000
BUFFER_HOURS = 12
WAVE_FILENAME = 'test/test2.wav'
dumps = {'svm': {'mfcc': '../models/svm_mfcc_22050.joblib.pkl',
                 'fbank': '../models/svm_fbank_22050.joblib.pkl'}}
feature_func = {'mfcc': mfcc, 'fbank': logfbank}


######################################################################
# etc function part
######################################################################
def d_mfcc(signal,samplerate=16000,winlen=0.025,winstep=0.01,numcep=13,nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,ceplifter=22,appendEnergy=True,winfunc=lambda x:numpy.ones((x,))):
    feat = mfcc(signal,samplerate,nfft=nfft,)
    d_feat = delta(feat, 2)
    return d_feat

def get_plots(audio_signal, current_pos, parameters, SAMPLE_RATE, CHUNK_SIZE):
    """
    return audio_signal and audio_plot after normalizing/smoothing
    audio_plot starts with [0.0, 0.0, ...] so that the latest readings
    are at the end
    """
    # roll the arrays so that the latest readings are at the end
    buffer_len = audio_signal.shape[0]
    audio_signal = np.roll(audio_signal, shift=buffer_len-current_pos)

    # normalise volume level
    audio_signal /= parameters['upper_limit']

    # apply some smoothing
    sigma = 4 * (SAMPLE_RATE / float(CHUNK_SIZE))
    audio_signal = ndimage.gaussian_filter1d(audio_signal, sigma=sigma, mode="reflect")

    # get the last hour of data for the plot and re-sample to 1 value per second
    hour_chunks = int(60 * 60 * (SAMPLE_RATE / float(CHUNK_SIZE)))
    xs = np.arange(hour_chunks)
    f = interpolate.interp1d(xs, audio_signal[-hour_chunks:])
    audio_plot = f(np.linspace(start=0, stop=xs[-1], num=3600))

    return [audio_signal, audio_plot]

def normalize(audio_signal, time_stamps, current_pos, parameters, SAMPLE_RATE, CHUNK_SIZE):
    # roll the arrays so that the latest readings are at the end
    buffer_len = time_stamps.shape[0]
    time_stamps = np.roll(time_stamps, shift=buffer_len-current_pos)
    audio_signal = np.roll(audio_signal, shift=buffer_len-current_pos)

    # normalise volume level
    audio_signal /= parameters['upper_limit']

    # apply some smoothing
    sigma = 4 * (SAMPLE_RATE / float(CHUNK_SIZE))
    audio_signal = ndimage.gaussian_filter1d(audio_signal, sigma=sigma, mode="reflect")

    # get the last hour of data for the plot and re-sample to 1 value per second
    hour_chunks = int(60 * 60 * (SAMPLE_RATE / float(CHUNK_SIZE)))
    xs = np.arange(hour_chunks)
    f = interpolate.interp1d(xs, audio_signal[-hour_chunks:])
    audio_plot = f(np.linspace(start=0, stop=xs[-1], num=3600))

    # ignore positions with no readings
    mask = time_stamps > 0
    time_stamps = time_stamps[mask]
    audio_signal = audio_signal[mask]
    results = [audio_plot, audio_signal, time_stamps]
    return results

def format_time_difference(time1, time2):
    time_diff = datetime.fromtimestamp(time2) - datetime.fromtimestamp(time1)
    return str(time_diff).split('.')[0]

# def process_signal(audio_plot, audio_signal, parameters, time_stamps):
#     # partition the audio history into blocks of type:
#     #   1. noise, where the volume is greater than noise_threshold
#     #   2. silence, where the volume is less than noise_threshold
#     noise = audio_signal > parameters['noise_threshold']
#     silent = audio_signal < parameters['noise_threshold']

#     # # join "noise blocks" that are closer together than min_quiet_time
#     # crying_blocks = []
#     # if np.any(noise):
#     #     silent_labels, _ = ndimage.label(silent)
#     #     silent_ranges = ndimage.find_objects(silent_labels)
#     #     for silent_block in silent_ranges:
#     #         start = silent_block[0].start
#     #         stop = silent_block[0].stop

#     #         # don't join silence blocks at the beginning or end
#     #         if start == 0:
#     #             continue

#     #         interval_length = time_stamps[stop-1] - time_stamps[start]
#     #         if interval_length < parameters['min_quiet_time']:
#     #             noise[start:stop] = True

#     #     # find noise blocks start times and duration
#     #     crying_labels, num_crying_blocks = ndimage.label(noise)
#     #     crying_ranges = ndimage.find_objects(crying_labels)
#     #     for cry in crying_ranges:
#     #         start = time_stamps[cry[0].start]
#     #         stop = time_stamps[cry[0].stop-1]
#     #         duration = stop - start

#     #         # ignore isolated noises (i.e. with a duration less than min_noise_time)
#     #         if duration < parameters['min_noise_time']:
#     #             continue

#     #         # save some info about the noise block
#     #         crying_blocks.append({'start': start,
#     #                               'start_str': datetime.fromtimestamp(start).strftime("%I:%M:%S %p").lstrip('0'),
#     #                               'stop': stop,
#     #                               'duration': format_time_difference(start, stop)})

#     # determine how long have we been in the current state
#     # time_current = time.time()
#     # time_crying = ""
#     # time_quiet = ""
#     # str_crying = "Drone noise for "
#     # str_quiet = "Drone quiet for "
#     # if len(crying_blocks) == 0:
#     #     time_quiet = str_quiet + format_time_difference(time_stamps[0], time_current)
#     # else:
#     #     if time_current - crying_blocks[-1]['stop'] < parameters['min_quiet_time']:
#     #         time_crying = str_crying + format_time_difference(crying_blocks[-1]['start'], time_current)
#     #     else:
#     #         time_quiet = str_quiet + format_time_difference(crying_blocks[-1]['stop'], time_current)


#     results = {'audio_plot': audio_plot,
#                # 'crying_blocks': crying_blocks,
#                # 'time_crying': time_crying,
#                # 'time_quiet': time_quiet
#                }
#     return results


######################################################################
# web server part
######################################################################
clients = []

class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.render('index.html')

class WebSocketHandler(tornado.websocket.WebSocketHandler):
    def open(self):
        print("New connection")
        clients.append(self)

    def on_close(self):
        print("Connection closed")
        clients.remove(self)

def broadcast_mic_data():
    # get the latest data from the audio server
    parameters = {"upper_limit": UPPER_LIMIT,
                  "noise_threshold": NOISE_THRESHOLD,
                  # "min_quiet_time": MIN_QUIET_TIME,
                  # "min_noise_time": MIN_NOISE_TIME
                  }
    conn = Client(AUDIO_SERVER_ADDRESS)
    conn.send(parameters)
    results = conn.recv()
    conn.close()
    # print("received results: ")
    # print(results)

    # send results to all clients
    now = datetime.now()
    results['date_current'] = '{dt:%A} {dt:%B} {dt.day}, {dt.year}'.format(dt=now)
    results['time_current'] = now.strftime("%I:%M:%S %p").lstrip('0')
    # results['audio_plot'] = results['audio_plot'].tolist()
    # print(results)
    for c in clients:
        c.write_message(results)


def web_server_init():
    print("web server start")
    settings = {
        "static_path": os.path.join(os.path.dirname(__file__), "static"),
    }
    app = tornado.web.Application(
        handlers=[
            (r"/", IndexHandler),
            (r"/ws", WebSocketHandler),
        ], **settings
    )
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(WEB_SERVER_ADDRESS[1], WEB_SERVER_ADDRESS[0])
    print("Listening on port:", WEB_SERVER_ADDRESS[1])
 
    main_loop = tornado.ioloop.IOLoop.instance()
    scheduler = tornado.ioloop.PeriodicCallback(broadcast_mic_data, 2000, io_loop=main_loop)
    scheduler.start()
    main_loop.start()


######################################################################
# audio server part
######################################################################
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

    listener = Listener(AUDIO_SERVER_ADDRESS)

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

            # print(audio)
            # print(np.average(audio))
            print(percent_detected)

            conn = listener.accept()
            conn.send({
                "wav": audio.flatten().tolist(),
                "percent_detected": percent_detected
            })
            conn.close()

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

def audio_server_init():
    print("audio server start")
    p1 = mp.Process(target=process_audio)
    p1.start()


######################################################################
# main
######################################################################
def main():
    audio_server_init()
    time.sleep(4)   # wait for a while
    web_server_init()

if __name__ == '__main__':
    print("server start")
    main()