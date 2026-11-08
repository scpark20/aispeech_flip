# Setup notes

## Audio exploration

Start Jupyter in `asr/` so the local audio and helper modules resolve correctly. `spectrogram.ipynb` reads the included `LJ001-0001.wav` file. It uses NumPy, Matplotlib, librosa, and IPython audio playback.

## Recognition data

The recognition notebooks use `KSponSpeechDataset(root_dir='/data/KsponSpeech')`. Change `root_dir` to your dataset location before creating the data loader.

The loader searches recursively for matching `.txt` and `.pcm` files. It reads transcripts as **CP949** and audio as **16-bit PCM at 16 kHz**, then computes 80-channel mel-spectrograms. The corpus is not included in this repository.

## Training environment

The model notebooks call `.cuda()` and set `CUDA_VISIBLE_DEVICES`. They require CUDA-capable PyTorch as written. The transducer also imports `warp_rnnt`; that extension must match the installed PyTorch/CUDA environment.

The saved notebook metadata records **Python 3.8.8**. There is no pinned dependency specification. Some librosa calls use older positional-argument APIs, so newer releases may require adaptation. A fresh end-to-end training run has not been verified as part of the documentation update.

## Checkpoint state

The training notebooks retain resume settings from earlier runs:

| Notebook | Save directory | Initial step |
| --- | --- | --- |
| CTC | `save/ctc_lstm_kspon` | 76019 |
| LAS | `save/las_lstm_kspon` | 30711 |
| Transducer | `save/transducer_lstm_kspon` | 43400 |

No checkpoints are bundled. For a fresh run, set `step = 0`, skip the `logger.load(...)` block, and use a fresh log directory. For a resumed run, provide the corresponding checkpoint and log files.

`Logger.load` returns without loading if its checkpoint path is absent; the notebook's following `loaded` print alone does not confirm restoration.
