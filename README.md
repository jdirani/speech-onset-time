# Speech onset detection toolkit

## Record speech and detect speech onset:
signal = record_voice()


## To detect utterance time (see EXAMPLE):
        # 1) Read the wav using wav.read()
        # 2) optionally LPF the wavs using FilterSignal()
        # 3) _get_envelope()
        # 4) _get_voice_onset()

        # NOTE: step (3) contains an LPF which is applied to the envelope. It
        # is not applied to the signal, so not a duplicate of step (2)
