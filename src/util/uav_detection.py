import numpy as np
import time
import ctypes
from scipy import ndimage, interpolate
from datetime import datetime

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

def process_signal(audio_plot, audio_signal, parameters, time_stamps):
    # partition the audio history into blocks of type:
    #   1. noise, where the volume is greater than noise_threshold
    #   2. silence, where the volume is less than noise_threshold
    noise = audio_signal > parameters['noise_threshold']
    silent = audio_signal < parameters['noise_threshold']

    # join "noise blocks" that are closer together than min_quiet_time
    crying_blocks = []
    if np.any(noise):
        silent_labels, _ = ndimage.label(silent)
        silent_ranges = ndimage.find_objects(silent_labels)
        for silent_block in silent_ranges:
            start = silent_block[0].start
            stop = silent_block[0].stop

            # don't join silence blocks at the beginning or end
            if start == 0:
                continue

            interval_length = time_stamps[stop-1] - time_stamps[start]
            if interval_length < parameters['min_quiet_time']:
                noise[start:stop] = True

        # find noise blocks start times and duration
        crying_labels, num_crying_blocks = ndimage.label(noise)
        crying_ranges = ndimage.find_objects(crying_labels)
        for cry in crying_ranges:
            start = time_stamps[cry[0].start]
            stop = time_stamps[cry[0].stop-1]
            duration = stop - start

            # ignore isolated noises (i.e. with a duration less than min_noise_time)
            if duration < parameters['min_noise_time']:
                continue

            # save some info about the noise block
            crying_blocks.append({'start': start,
                                  'start_str': datetime.fromtimestamp(start).strftime("%I:%M:%S %p").lstrip('0'),
                                  'stop': stop,
                                  'duration': format_time_difference(start, stop)})

    # determine how long have we been in the current state
    time_current = time.time()
    time_crying = ""
    time_quiet = ""
    str_crying = "Drone noise for "
    str_quiet = "Drone quiet for "
    if len(crying_blocks) == 0:
        time_quiet = str_quiet + format_time_difference(time_stamps[0], time_current)
    else:
        if time_current - crying_blocks[-1]['stop'] < parameters['min_quiet_time']:
            time_crying = str_crying + format_time_difference(crying_blocks[-1]['start'], time_current)
        else:
            time_quiet = str_quiet + format_time_difference(crying_blocks[-1]['stop'], time_current)


    results = {'audio_plot': audio_plot,
               'crying_blocks': crying_blocks,
               'time_crying': time_crying,
               'time_quiet': time_quiet}
    return results
