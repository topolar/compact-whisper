# compact-whisper
This project is based on script [whisper-standalone-win](https://github.com/Purfview/whisper-standalone-win/blob/main/faster-whisper%20cli/__main__.py). Writters were removed
and refactored to support max_line_width and max_line_count parameters. Project is using `faster-whisper` for file processing.

It can be used directly via `compact-whisper.py` script or as standalone executable.

## Requirements to run on GPU -  CUDA
You need GFX with support for CUDA,CUDNN. Tested on NVIDIA GeForce RTX 2080 and 3060.
- [CUDA](https://developer.nvidia.com/cuda-11.2.0-download-archive)
- [CUDNN](https://developer.nvidia.com/cudnn) - requires dev registration
- `zlibwapi.dll` (Required by CUDNN but not included!)

## To run on CPU:
use parameter `--device cpu`

## Example of usage
Usage is similar as use of `whisper` or `whisper-ctranslate2`. Use `--help` to see all parameters.

`compact-whisper.exe data\sample01.wav --language cs --vad_filter True --model base --word_timestamps True --output_format all --max_line_width 42 --max_line_count 2`

### Supported models
|  Size  | Parameters | English-only model | Multilingual model | Required VRAM | Relative speed |
|:------:|:----------:|:------------------:|:------------------:|:-------------:|:--------------:|
|  tiny  |    39 M    |     `tiny.en`      |       `tiny`       |     ~1 GB     |      ~32x      |
|  base  |    74 M    |     `base.en`      |       `base`       |     ~1 GB     |      ~16x      |
| small  |   244 M    |     `small.en`     |      `small`       |     ~2 GB     |      ~6x       |
| medium |   769 M    |    `medium.en`     |      `medium`      |     ~5 GB     |      ~2x       |
| large  |   1550 M   |        N/A         |      `large`       |    ~10 GB     |       1x       |


## Create standalone executable by pyinstaller
`pyinstaller -F compact-whisper.py --collect-all faster_whisper  --exclude-module torch`

## Example of start with script:
`python .\compact-whisper.py data\sample01.wav --language cs --vad_filter True --model base --word_timestamps True --output_format all --max_line_width 42 --max_line_count 2`