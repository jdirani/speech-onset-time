# Contains various signal processing functions

# To detect utterance time (see EXAMPLE):
        # 1) Read the wav using wav.read()
        # 2) optionally LPF the wavs using FilterSignal()
        # 3) get_envelope()
        # 4) get_voice_onset()

        # NOTE: step (3) contains an LPF which is applied to the envelope. It
        # is not applied to the signal, so not a duplicate of step (2)


import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from matplotlib import pyplot as plt
import numpy,os, csv
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt



def record_voice(out_name=None, channels=1, fs=44100, duration=2, dtype='float64'):
    '''
    out_name: if not None, writes to file.

    channels: 1 for mono, 2 for stereo

    duration: in seconds

    dtype: 'int16' for 16 bit, float64 for 32 bit rate.

    fs: sampling frequency
    '''
    myrecording = sd.rec(duration * fs, samplerate=fs, channels=channels,dtype=dtype)
    sd.wait()
    if out_name: # Writes to file
        wav.write('%s.wav'%out_name, data=myrecording, rate=fs)

    return myrecording



#--- Based on Jarne (2017) "Simple empirical algorithm to obtain signal envelope in three steps"
def get_envelope(signal, fs=44100, N=200, cutoff=2000):
    '''
    signal: input wav (numpy.ndarray)

    fs: sampling frequency

    N: number of samples per chunk (in part (2))

    cutoff: LPF cutoff, the smaller the cuttoff the stronger the filter. (tweek this).
    '''
    # 1) Take the absolute value of the signal
    abs_signal = abs(signal)
    # 2) Seperate into samples of N, and get peak value of each sample.
    chunked_signal = [abs_signal[i:i+N] for i in range(0, len(abs_signal), N)]
    new_signal = []
    for chunk in chunked_signal: #Then for each chunk, replace all values by max value
        max_value = np.max(chunk)
        new_chunk = [max_value for i in range(len(chunk))]
        new_signal.append(new_chunk)
    new_signal = np.array(new_signal).flatten()
    # 3) LPF the new_signal (the envelope, not the original signal)
    def FilterSignal(signal_in, fs, cutoff):
        B, A = butter(1, cutoff / (fs / 2.0), btype='low')
        filtered_signal = filtfilt(B, A, signal_in, axis=0)
        return filtered_signal
    filteredSignal = FilterSignal(new_signal, fs, cutoff)

    return filteredSignal




# Note The threshold depends also on the input volume set on the computer
def get_voice_onset(signal, threshold = 200, fs=44100, n_above=441):
    '''
    signal: signal in = the envelope (numpy.ndarray)

    threshold : amplitude threshold for voice onset. threshold = 200 with MEG mic at 75% input volume seems to work well.

    fs: sampling frequency

    n_above: number of samples after the threshold is crossed used to calculate
             the median amplitude and decide if it was random burst of noise
            or speech onset.
    '''
    indices_onset = np.where(signal >= threshold)[0] # All indices above threshold
    # Next, find the first index that where the MEDIAN stays above threshold for the next 10ms
    # Not using the MEAN because sensitive to a single extreme value
    # Note 44.1 points per millesconds (for fs=44100)
    # 10ms = 441 points
    for i in indices_onset:
        # avg_10ms = np.array([abs(j) for j in signal[i:i+n_above]]).flatten().mean()
        avg_10ms = np.median(np.array([abs(j) for j in signal[i:i+n_above]]).flatten())
        if avg_10ms >= threshold:
            idx_onset = i
            onset_time = idx_onset / float(fs) * 1000.0

            return idx_onset, onset_time



# LPF function: used in get_envelope() but here for seperate use.
def FilterSignal(signal_in, fs=44100, cutoff=200):
    '''
    signal_in: input wav (numpy.ndarray)

    fs: sampling frequency

    cutoff: LPF cutoff. 200 works well with MEG mic
    '''
    B, A = butter(1, cutoff / (fs / 2.0), btype='low')
    filtered_signal = filtfilt(B, A, signal_in, axis=0)
    return filtered_signal


def auto_utterance_times(dir_in, file_out):
    '''
    Automatically get utterance times. Output results to file.
    '''
    voices = [i for i in os.listdir(dir_in) if i.endswith('.wav')]
    rts_out = []

    for v in voices:
        signal = wav.read(os.path.join(dir_in + '/'+ v))[1]
        flt_signal = FilterSignal(signal)
        env = get_envelope(flt_signal)
        idx, rt = get_voice_onset(env)
        rts_out.append(rt)

    f = open(file_out,'w')
    for r in rts_out:
        f.write('%s\n'%r)
    f.close()



def plot_auto_utterance_times(dir_in, dir_out, fs=44100):
    '''
    Plot automatically generated utterance times.
    '''
    voices = [i for i in os.listdir(dir_in) if i.endswith('.wav')]

    for v in voices:
        signal = wav.read(os.path.join(dir_in + '/'+ v))[1]
        flt_signal = FilterSignal(signal)
        env = get_envelope(flt_signal)
        idx, rt = get_voice_onset(env)

        # Creating X axis, in miliiseconds
        N_samples = len(signal)
        len_signal_ms = len(signal)/fs*1000
        X_axis = np.arange(0,len_signal_ms,1.0/fs*1000)

        #plotting
        fig, ax = plt.subplots(figsize=((18,5)))
        ax.plot(X_axis, signal, color='b')
        ax.axvline(rt, color='r')
        plt.savefig(os.path.join(dir_out + '/' + v[:-4] + '.jpg'))
        plt.close()



def semi_auto_utterance_times(dir_in, dir_out, fs=44100):
    '''
    Returns list of RTs and saves plots to dir_out.
    '''
    plt.ion()
    def onpick(event):
        if event.dblclick:
            manual_rts.append((event.xdata, event.xdata/fs)) #manual_rts is a variable defined in below semi_auto_utterance_times. Not the best way to write this.
            L =  ax.axvline(x=event.xdata, color='orange')
            fig.canvas.draw()
        elif event.button == 3:
            plt.close()
            fig.canvas.mpl_disconnect(cid)
            return

    voices = [i for i in os.listdir(dir_in) if i.endswith('.wav')]
    rts_out = []

    nb = 0 #keep count
    for v in voices:
        nb+=1
        print '#%s'%nb, v
        # auto fetch RT
        signal = wav.read(os.path.join(dir_in + '/'+ v))[1]
        flt_signal = FilterSignal(signal)
        env = get_envelope(flt_signal)
        rt_auto = get_voice_onset(env) #rt_auto (idx,rt)
        print "rt auto ", rt_auto

        # Creating X axis, in miliiseconds
        N_samples = len(signal)
        len_signal_ms = len(signal)/fs*1000
        X_axis = np.arange(0,len_signal_ms,1.0/fs*1000)

        # plot and fix rt
        manual_rts = [] # DEFINED HERE manual_rts
        fig, ax = plt.subplots(figsize=((18,5)))
        ax.plot(X_axis, signal, color='b')
        if rt_auto:
            ax.axvline(rt_auto[1], color='r')
        cid = fig.canvas.mpl_connect('button_press_event', onpick)
        raw_input('press enter to continue...') #pauses to wait for cid to finish

        # print "rt manual ", rt_manual
        # print "manual_rts (%s)"%len(manual_rts), manual_rts

        if len(manual_rts) > 0:
            rt = (manual_rts[-1][0], manual_rts[-1][1]*1000) #Keep idx as is, convert rt to milliseconds
        else:
            rt = rt_auto

        print 'final rt ', rt
        print '--------------------'
        rts_out.append(rt)

        plt.savefig(os.path.join(dir_out + '/' + v[:-4] + '.jpg'))
        plt.close()

    return rts_out


# Returns a dict with trialnum, target, and rt
def semi_auto_utterance_times_PorthalFormat(dir_in, dir_out, fs=44100):
    '''
    Returns list of RTs and csv with RTs and trial info, and saves plots to dir_out.
    '''
    plt.ion()
    def onpick(event):
        if event.dblclick:
            manual_rts.append((event.xdata, event.xdata/fs)) #manual_rts is a variable defined in below semi_auto_utterance_times. Not the best way to write this.
            L =  ax.axvline(x=event.xdata, color='orange')
            fig.canvas.draw()
        elif event.button == 3:
            plt.close()
            fig.canvas.mpl_disconnect(cid)
            return

    voices = [i for i in os.listdir(dir_in) if i.endswith('.wav')]
    trials_rts = []

    nb = 0 #keep count
    for v in voices:
        nb+=1
        print '#%s'%nb, v
        # auto fetch RT
        signal = wav.read(os.path.join(dir_in + '/'+ v))[1]
        flt_signal = FilterSignal(signal)
        env = get_envelope(flt_signal)
        rt_auto = get_voice_onset(env) #rt_auto (idx,rt)
        print "rt auto ", rt_auto

        # Creating X axis, in miliiseconds
        N_samples = len(signal)
        len_signal_ms = len(signal)/fs*1000
        X_axis = np.arange(0,len_signal_ms,1.0/fs*1000)

        # plot and fix rt
        manual_rts = [] # DEFINED HERE manual_rts
        fig, ax = plt.subplots(figsize=((18,5)))
        ax.plot(X_axis, signal, color='b')
        if rt_auto:
            ax.axvline(rt_auto[1], color='r')
        cid = fig.canvas.mpl_connect('button_press_event', onpick)
        raw_input('press enter to continue...') #pauses to wait for cid to finish

        # print "rt manual ", rt_manual
        # print "manual_rts (%s)"%len(manual_rts), manual_rts

        if len(manual_rts) > 0:
            rt = (manual_rts[-1][0], manual_rts[-1][1]*1000) #Keep idx as is, convert rt to milliseconds
        else:
            rt = rt_auto #convert to milliseconds

        print 'final rt ', rt
        print '--------------------'


        if not os.path.isdir(os.path.join(dir_out, 'rts')):
            os.makedirs(os.path.join(dir_out, 'rts'))

        plt.savefig(os.path.join(dir_out,'rts', (v[:-4]+'.jpg')))
        plt.close()

        trialnum = v.split('_')[0]
        target = v.split('_')[1][:-4]
        trials_rts.append({'target':target,'trialID':trialnum,'rt':rt[1]})

        with open(os.path.join(dir_out,'rts','RT_out.csv'), 'w') as f:
            w = csv.DictWriter(f, fieldnames=['trialID', 'target','rt'])
            w.writeheader()
            w.writerows(trials_rts)


    return trials_rts
