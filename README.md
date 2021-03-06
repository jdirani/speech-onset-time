# Speech-Onset Detection Toolkit
This toolkit was developed in order to easily detect the onset of speech in voice recordings for psychology experiments.

## Record speech
signal = record_voice()


## Automatically detect utterance time:
**auto_utterance_times():** To return the onset time of a single .wav file <br>
**auto_utterance_times_mult():** Takes a directory of .wav as input and returns speech onset time for all audio files.<br>
**plot_auto_utterance_times():** Plot results of auto_utterance_times_mult() to monitor accuracy,

## Semi-automatic detection of utterance time:
**semi_auto_utterance_times():** Automatically detects utterance times and plots them, allowing to manually edit the predictions if needed. <br>
To edit prediction, double click on plot to move the vertical line. Press enter on terminal to go to the next plot.
Figures are saved to file.



