# compact-whisper

Based on whisper-standalone-win

## Create standalone executables of OpenAI's Whisper & Faster-Whisper
`pyinstaller -F compact-whisper.py`

Standalone executables of [OpenAI's Whisper](https://github.com/openai/whisper) & [Faster-Whisper](https://github.com/guillaumekln/faster-whisper) for those who don't want to bother with Python.

Executables are compatible with Windows 7 x64 and above.    
Meant to be used in command-line interface or [Subtitle Edit](https://github.com/SubtitleEdit/subtitleedit).   
Faster-Whisper is much faster than OpenAI's Whisper, and it requires less RAM/VRAM.

## Usage examples
* `whisper.exe "D:\videofile.mkv" --language=English --model=medium`   

* `whisper.exe --help`

## Notes

Run your command-line interface as Administrator.   
Don't copy programs to the Windows' folders!   
By default the subtitles are created in the same folder where an executable file is located.   
Programs automatically will choose to work on GPU if CUDA is detected.   
For decent transcription use not smaller than `medium` model.   
Guide how to run the command line programs: https://www.youtube.com/watch?v=A3nwRCV-bTU
   
## OpenAI's Whisper standalone info

OpenAI version needs 'FFmpeg.exe' in PATH, or copy it to Whisper's folder [Subtitle Edit downloads FFmpeg automatically].
   
   
## Faster-Whisper standalone info

Some defaults are tweaked for movies transcriptions and to make it portable.   
By default it looks for models in the same folder, in path like this -> `_models\faster-whisper-medium`.   
Models are downloaded automatically or can be downloaded manually from: https://huggingface.co/guillaumekln   
In Subtitle Edit it can be selected for CTranslate2 engine, just rename it to `whisper-ctranslate2.exe`.   
"large" is mapped to `large-v2` model.
