# Speech onset detection toolkit

## Record speech
signal = record_voice()


## Automarically detect utterance time:
auto_utterance_times(): To return the onset time of a single .wav file <br>
auto_utterance_times_mult(): Takes a directory of .wav as input and returns speech onset time for all audio files.
plot_auto_utterance_times(): Plot results of auto_utterance_times_mult() to monitor accuracy,

## Semi-automatic detecttion of utterance time:
semi_auto_utterance_times(): Automatically detects utterance times and plots them, allowing to manually edit the predictions if needed.

