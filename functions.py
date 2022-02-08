import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import numpy, os, csv
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt
from tqdm import tqdm

def record_voice(out_name=None, channels=1, fs=44100, duration=2, dtype='float64'):
    '''
    out_name : str (directory)
               If not None, writes to file.

    channels: 1 for mono, 2 for stereo

    duration : In seconds

    dtype : 'int16' for 16 bit, float64 for 32 bit rate.

    fs : int
         Sampling frequency
    '''
    myrecording = sd.rec(duration * fs, samplerate=fs, channels=channels,dtype=dtype)
    sd.wait()
    if out_name: # Writes to file
        wav.write('%s.wav'%out_name, data=myrecording, rate=fs)

    return myrecording



#--- Based on Jarne (2017) "Simple empirical algorithm to obtain signal envelope in three steps"
def _get_envelope(signal, fs=44100, N=200, cutoff=2000):
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
    # new_signal = np.array(new_signal).flatten()
    new_signal = np.array([item for sublist in new_signal for item in sublist]) # flatten list of lists
    # 3) LPF the new_signal (the envelope, not the original signal)
    def FilterSignal(signal_in, fs, cutoff):
        B, A = butter(1, cutoff / (fs / 2.0), btype='low')
        filtered_signal = filtfilt(B, A, signal_in, axis=0)
        return filtered_signal
    filteredSignal = FilterSignal(new_signal, fs, cutoff)

    return filteredSignal




# Note The threshold depends also on the input volume set on the computer
def _get_voice_onset(signal, threshold = 200, fs=44100, time_above_thresh=100):
    '''
    signal : numpy.ndarray
             signal in. Should be the envelope of the raw signal for accurate results

    threshold : int
                Amplitude threshold for voice onset.
                (Threshold = 200 with NYUAD MEG mic at 75% input volume seems to work well)

    fs : int
         Sampling frequency

    tim_above_thresh : int (ms)
             Time in ms after the threshold is crossed used to calculate
              the median amplitude and decide if it was random burst of noise
              or speech onset.
    '''

    n_above_thresh = int(fs/time_above_thresh) # convert time above threshold to number of samples.

    indices_onset = np.where(signal >= threshold)[0] # All indices above threshold
    # Next, find the first index that where the MEDIAN stays above threshold for the next 10ms
    # Not using the MEAN because sensitive to a single extreme value
    # Note 44.1 points per millesconds (for fs=44100)
    # 10ms = 441 points
    for i in indices_onset:
        # avg_10ms = np.array([abs(j) for j in signal[i:i+n_above_thresh]]).flatten().mean()
        avg_10ms = np.median(np.array([abs(j) for j in signal[i:i+n_above_thresh]]).flatten())
        if avg_10ms >= threshold:
            idx_onset = i
            onset_time = idx_onset / float(fs) * 1000.0

            return idx_onset, onset_time
    return np.nan, np.nan # if no point exceeds the threshold.
                          # Return "None" instead of None in order to be able to append it to a list later on



# LPF function: used in _get_envelope() but here for seperate use.
def FilterSignal(signal_in, fs=44100, cutoff=200):
    '''
    signal_in : numpy.ndarray
                Input .wav signal

    fs : int
         Sampling frequency

    cutoff : int
             LPF cutoff (200 works well with NYUAD MEG mic)
    '''
    B, A = butter(1, cutoff / (fs / 2.0), btype='low')
    filtered_signal = filtfilt(B, A, signal_in, axis=0)
    return filtered_signal



def auto_utterance_times(signal, fs, threshold=200, time_above_thresh=100):
    '''
    Automatically get utterance times.
    Returns idx and rt (ms)

    signal, fs : np.array, int
                 fs, signal = wav.read(file_in)

    threshold, time_above_thresh : see _get_voice_onset()

    Returns
    ---------
    idx : index of utterance time
    rt : in milliseconds

    '''
    flt_signal = FilterSignal(signal,fs=fs)
    env = _get_envelope(flt_signal,fs=fs)
    idx, rt = _get_voice_onset(env,fs=fs, threshold=threshold, time_above_thresh=time_above_thresh)

    return idx, rt


def plot_utterance_time(signal, fs, rt, title=None, show=True, savefig=None):
    '''
    signal, fs : np.array, int
                 fs, signal = wav.read(file_in)

    rt : float
         rt in milliseconds

    show : bool
           Toggle show figure

    savefig : None, str
              Path to save figure. If None, figure is not saved
    '''

    # Creating X axis, in miliiseconds
    N_samples = len(signal)
    len_signal_ms = len(signal)/fs*1000
    times = np.arange(0,len_signal_ms,1.0/fs*1000)

    #plotting
    fig, ax = plt.subplots(figsize=((18,5)))
    ax.plot(times, signal, color='b')
    ax.axvline(rt, color='r')
    ax.set_title(title)

    if savefig:
        plt.savefig(savefig)

    if show:
        plt.show()

    return fig, ax



def auto_utterance_times_batch(wav_paths, output_txt=None, plots_dir=None, threshold=200, time_above_thresh=100):
    '''
    Automatically get utterance times. Returns list of indices and rts (ms).
    Option to output results to file.

    wav_paths : list paths to .wav files

    threshold, time_above_thresh : see _get_voice_onset()

    output_txt : None, str
                 Path to txt file output

    plots_dir : None, str
                Directory to plot waveform and utterance times. If none, no plotting done.

    Returns
    ---------
    indices : indices of utterance times
    rts : in milliseconds

    '''


    rts_out = []
    idx_out = []

    for v in tqdm(wav_paths):
        fs, signal = wav.read(v)
        flt_signal = FilterSignal(signal,fs=fs)
        env = _get_envelope(flt_signal,fs=fs)
        idx, rt = _get_voice_onset(env,fs=fs, threshold=threshold, time_above_thresh=time_above_thresh)
        rts_out.append(rt)
        idx_out.append(idx)

        if plots_dir:
            title = os.path.basename(v).replace('.wav','')
            fig_path = os.path.join(plots_dir, title+'.jpg')
            plot_utterance_time(signal, fs, rt, title=title, show=False, savefig=fig_path)
            plt.close()

    if output_txt:
        f = open(output_txt,'w')
        for r in rts_out:
            f.write('%s\n'%r)
        f.close()

    return idx_out, rts_out




def semi_auto_utterance_times(dir_in, dir_out):
    '''

    Automatically detects utterance times and plots them, allowing to manually edit the prediction.
    To edit prediction, double click on plot to move the vertical line. Press enter on terminal to go to the next plot.
    Figures are saved to file.


    dir_in : str (directory)
            Directory containing .wav files

    dir_out : str (directory)
            Output directory


    '''

    plt.ion()
    def onpick(event):
        if event.dblclick:
            manual_rts.append(event.xdata) #manual_rts is a variable defined in below semi_auto_utterance_times. Not the best way to write this.
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
        print('#%s'%nb, v)
        # auto fetch RT
        fs, signal = wav.read(os.path.join(dir_in, v))
        flt_signal = FilterSignal(signal)
        env = _get_envelope(flt_signal)
        rt_auto = _get_voice_onset(env) #rt_auto (idx,rt)
        print("rt auto ", rt_auto)

        # Creating X axis, in miliiseconds
        N_samples = len(signal)
        len_signal_ms = len(signal)/fs*1000
        X_axis = np.arange(0,len_signal_ms,1.0/fs*1000)

        # plot and fix rt
        manual_rts = [] # DEFINED HERE manual_rts
        fig, ax = plt.subplots(figsize=((18,5)))
        plt.title(v)
        ax.plot(X_axis, signal, color='b')
        if rt_auto:
            ax.axvline(rt_auto[1], color='r')
        cid = fig.canvas.mpl_connect('button_press_event', onpick)
        input('press enter to continue...') #pauses to wait for cid to finish

        # print("rt manual ", rt_manual)
        # print("manual_rts (%s)"%len(manual_rts), manual_rts)

        if len(manual_rts) > 0:
            # rt = (manual_rts[-1][0], manual_rts[-1][1]*1000) #Keep idx as is, convert rt to milliseconds. commented out because idx=milliseconds now.
            rt = manual_rts[-1]
        else:
            rt = rt_auto[1] #convert to milliseconds

        print('final rt ', rt)
        print('--------------------')
        rts_out.append(rt)

        plt.savefig(os.path.join(dir_out, v[:-4] + '.jpg'))
        plt.close()

    return rts_out
