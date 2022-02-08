# Speech-Onset Detection Toolkit
This toolkit was developed in order to easily detect the onset of speech in voice recordings for psychology experiments.

## Automatically detect utterance time:
**auto_utterance_times():** To return the onset time of a single .wav file <br>
**auto_utterance_times_batch:** Takes list of paths (.wav) as input and returns speech onset time for all audio files. Option to plot the results<br>
**plot_utterance_time():** Plot results to monitor accuracy.

## Semi-automatic detection of utterance time:
**semi_auto_utterance_times():** Automatically detects utterance times and plots them, allowing to manually edit the predictions if needed. <br>
To edit prediction, double click on plot to move the vertical line. Press enter on terminal to go to the next plot.
Figures are saved to file.


## Example
    fs, signal = wav.read('path_in.wav')
    idx, rt = auto_utterance_times(signal, fs)
    plot_utterance_time(signal, fs, rt)
