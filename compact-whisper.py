import argparse
import json
import logging
import os.path
import sys
import typing

import ctranslate2
import faster_whisper
from faster_whisper.transcribe import WhisperModel
from faster_whisper.utils import download_model, format_timestamp
from faster_whisper.version import __version__

import numpy as np
import psutil
import time

import io
from os import system
from tqdm import tqdm
from threading import Thread
from time import sleep
from config import *
from writers import *
from version import __appver__


start_time = time.time()
default_model_dir = os.getcwd() + "/_models"




def from_language_to_iso_code(language):
    if language is not None:
        language_name = language.lower()
        if language_name not in LANGUAGES:
            if language_name in TO_LANGUAGE_CODE:
                language = TO_LANGUAGE_CODE[language_name]

    return language


def optional_int(string):
    return None if string == "None" else int(string)


def optional_float(string):
    return None if string == "None" else float(string)


def str2bool(string):
    str2val = {"True": True, "False": False, "true": True, "false": False}
    if string in str2val:
        return str2val[string]
    else:
        raise ValueError(f"Expected one of {set(str2val.keys())}, got {string}")



sys.stdout.reconfigure(encoding='utf-8')

class Unbuffered(TextIO):
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def writelines(self, datas):
        self.stream.writelines(datas)
        self.stream.flush()
    def getattr(self, attr):
        return getattr(self.stream, attr)
    def flush(self):
        self.stream.flush()


system_encoding = sys.getdefaultencoding()
stdout_encoding = sys.stdout.encoding


if system_encoding != "utf-8":
    def make_safe(string):
        return string.encode(system_encoding, errors="replace").decode(system_encoding)
elif stdout_encoding != "utf-8":
    def make_safe(string):
        return string.encode(stdout_encoding, errors="replace").decode(stdout_encoding)
else:
    def make_safe(string):
        return string

sys.stdout = Unbuffered(sys.stdout)


def cli():
    print(f"\ncompact-whisper {__appver__} by mistercz")
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("audio", nargs="+", type=str, help="audio file(s) to transcribe")
    parser.add_argument("--model", default="medium", help="name of the Whisper model to use")
    parser.add_argument("--model_dir", type=str, default=None, help=f"the path to save model files; uses {default_model_dir} by default")
    parser.add_argument("--device", default="cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu", help="device to use")
    parser.add_argument("--output_dir", "-o", type=str, default=".", help="directory to save the outputs")
    parser.add_argument("--output_format", "-f", type=str, default="srt", choices=["txt", "vtt", "srt", "tsv", "json", "all"], help="format of the output file; if not specified srt will be produced")
    parser.add_argument("--verbose", type=str2bool, default=False, help="whether to print out the progress and debug messages")
    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"], help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')")
    parser.add_argument("--language", type=str, default=None, choices=sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]), help="language spoken in the audio, specify None to perform language detection")
    parser.add_argument("--temperature", type=float, default=0, help="temperature to use for sampling")
    parser.add_argument("--best_of", type=optional_int, default=5, help="number of candidates when sampling with non-zero temperature")
    parser.add_argument("--beam_size", type=optional_int, default=1, help="number of beams in beam search, only applicable when temperature is zero")
    parser.add_argument("--patience", type=float, default=1.0, help="optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search")
    parser.add_argument("--length_penalty", type=float, default=1.0, help="optional token length penalty coefficient (alpha) as in https://arxiv.org/abs/1609.08144, uses simple length normalization by default")
    parser.add_argument("--suppress_tokens", type=str, default="-1", help="comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations")
    parser.add_argument("--initial_prompt", type=str, default=None, help="optional text to provide as a prompt for the first window.")
    parser.add_argument("--condition_on_previous_text", type=str2bool, default=True, help="if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop")
    parser.add_argument("--temperature_increment_on_fallback", type=optional_float, default=0.2, help="temperature to increase when falling back when the decoding fails to meet either of the thresholds below")
    parser.add_argument("--compression_ratio_threshold", type=optional_float, default=2.4, help="if the gzip compression ratio is higher than this value, treat the decoding as failed")
    parser.add_argument("--logprob_threshold", type=optional_float, default=-1.0, help="if the average log probability is lower than this value, treat the decoding as failed")
    parser.add_argument("--no_speech_threshold", type=optional_float, default=0.6, help="if the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence")
    parser.add_argument("--word_timestamps", type=str2bool, default=False, help="(experimental) extract word-level timestamps and refine the results based on them")
    parser.add_argument("--prepend_punctuations", type=str, default="\"\'“¿([{-", help="if word_timestamps is True, merge these punctuation symbols with the next word")
    parser.add_argument("--append_punctuations", type=str, default="\"\'.。,，!！?？:：”)]}、", help="if word_timestamps is True, merge these punctuation symbols with the previous word")
    parser.add_argument("--threads", type=optional_int, default=0, help="number of threads used for CPU inference; By default number of the real cores but no more that 4")
    parser.add_argument("--version", action="version", version="%(prog)s {version}".format(version=__version__), help="Show Faster-Whisper's version number")
    parser.add_argument("--vad_filter", type=str2bool, default=True, help="Enable the voice activity detection (VAD) to filter out parts of the audio without speech.")
    parser.add_argument("--vad_threshold", type=optional_float, default=0.45, help="Probabilities above this value are considered as speech.")
    parser.add_argument("--vad_min_speech_duration_ms", type=optional_int, default=350, help="Final speech chunks shorter min_speech_duration_ms are thrown out.")
    parser.add_argument("--vad_max_speech_duration_s", type=optional_float, default=None, help="Maximum duration of speech chunks in seconds. Longer will be split at the timestamp of the last silence.")
    parser.add_argument("--vad_min_silence_duration_ms", type=optional_int, default=3000, help="In the end of each speech chunk time to wait before separating it.")
    parser.add_argument("--vad_speech_pad_ms", type=optional_int, default=900, help="Final speech chunks are padded by speech_pad_ms each side.")
    parser.add_argument("--vad_window_size_samples", type=optional_int, default=1024, help="Size of audio chunks fed to the silero VAD model. Values other than 512, 1024, 1536 may affect model perfomance!!!")
    parser.add_argument("--compute_type", type=str, default="default", choices=["default", "auto", "int8", "int8_float16", "int16", "float16", "float32"], help="Type of quantization to use (see https://opennmt.net/CTranslate2/quantization.html)")
    parser.add_argument("--max_line_width", type=optional_int, default=None, help="(requires --word_timestamps True) the maximum number of characters in a line before breaking the line")
    parser.add_argument("--max_line_count", type=optional_int, default=None, help="(requires --word_timestamps True) the maximum number of lines in a segment")
 

    args = parser.parse_args()

    if args.threads == 0:
        if psutil.cpu_count(logical=False) > 3:
            cores = 4
        else:
            cores = psutil.cpu_count(logical=False)
    else:
        cores = args.threads

    mod_el = args.model
    if args.model == "large":
        mod_el = "large-v2"
    model_dir = os.path.join((args.model_dir or default_model_dir), "faster-whisper-" + mod_el)

    chk_path1 = os.path.exists(os.path.join(model_dir, "config.json"))
    chk_path2 = os.path.exists(os.path.join(model_dir, "model.bin"))
    chk_path3 = os.path.exists(os.path.join(model_dir, "tokenizer.json"))
    chk_path4 = os.path.exists(os.path.join(model_dir, "vocabulary.txt"))

    if chk_path1 and chk_path2 and chk_path3 and chk_path4:
        model_path = model_dir
    else:
        if args.model in model_map:
            print("\nModel not found at: " + model_dir)
            print("Attempting to download:\n")
            model_path = download_model(mod_el, model_dir)
        else:
            print("\nUnknown model not found at: " + model_dir)
            print("\nTry these known models: ")
            print(list(model_map.keys()))
            sys.exit()

    device = args.device
    device_index = 0
    compute_type = args.compute_type
    if device.startswith('cuda:'):
        device, device_index = device.split(':')
        device_index = int(device_index)
    elif compute_type == "default":
        compute_type = "int8"

    temperature = args.temperature
    if (increment := args.temperature_increment_on_fallback) is not None:
        temperature = tuple(np.arange(temperature, 1.0 + 1e-6, increment))
    else:
        temperature = [temperature]

    language = from_language_to_iso_code(args.language)

    if (args.model.endswith(".en") and language != "en"):
        if args.language is not None:
            print("\nError: English model is selected when language is not English.")
            sys.exit()

    suppress_tokens = [int(t) for t in args.suppress_tokens.split(",")]

    if device.startswith('cuda'):
        print(f"running on: CUDA")
        print("Number of visible GPU devices: %s " % ctranslate2.get_cuda_device_count())
        if args.verbose:
            print("Supported compute types by GPU: %s \n" % ctranslate2.get_supported_compute_types("cuda", device_index))
    elif device.startswith('cpu'):
        print(f"running on: CPU")
        if args.verbose:
            print("Supported compute types by CPU: %s \n" % ctranslate2.get_supported_compute_types("cpu"))

    if args.verbose:
        logger = faster_whisper.utils.get_logger()
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        ctranslate2.set_log_level(logging.DEBUG)

    model = WhisperModel(model_path, device=device, device_index=device_index, compute_type=compute_type, cpu_threads=cores, download_root=default_model_dir)
    print("\nModel loaded in: %s seconds" % round((time.time() - start_time), 2))

    word_options = ["highlight_words", "max_line_count", "max_line_width"]
    argsDict=vars(args)
    if not args.word_timestamps:
        for option in word_options:
            if argsDict[option]:
                parser.error(f"--{option} requires --word_timestamps True")
    if args.max_line_count and not args.max_line_width:
        parser.warn("--max_line_count has no effect without --max_line_width")
    writer_args = {arg: argsDict.pop(arg,None) for arg in word_options}
    writer = get_writer(args.output_format, args.output_dir)

    vad_parameters = {}
    if args.vad_min_silence_duration_ms is not None:
        vad_parameters['min_silence_duration_ms'] = args.vad_min_silence_duration_ms
    if args.vad_min_speech_duration_ms is not None:
        vad_parameters['min_speech_duration_ms'] = args.vad_min_speech_duration_ms
    if args.vad_max_speech_duration_s is not None:
        vad_parameters['max_speech_duration_s'] = args.vad_max_speech_duration_s
    if args.vad_threshold is not None:
        vad_parameters['threshold'] = args.vad_threshold
    if args.vad_speech_pad_ms is not None:
        vad_parameters['speech_pad_ms'] = args.vad_speech_pad_ms
    if args.vad_window_size_samples is not None:
        vad_parameters['window_size_samples'] = args.vad_window_size_samples
    if not vad_parameters:
        vad_parameters = None

    start_time2 = time.time()
    for audio_path in args.audio:
        segments, info = model.transcribe(
            audio_path, language=language, task=args.task, beam_size=args.beam_size, best_of=args.best_of, patience=args.patience, length_penalty=args.length_penalty,
            temperature=temperature, compression_ratio_threshold=args.compression_ratio_threshold, log_prob_threshold=args.logprob_threshold,
            no_speech_threshold=args.no_speech_threshold, condition_on_previous_text=args.condition_on_previous_text, initial_prompt=args.initial_prompt,
            word_timestamps=args.word_timestamps, prepend_punctuations=args.prepend_punctuations, append_punctuations=args.append_punctuations,
            vad_filter=args.vad_filter, vad_parameters=vad_parameters, suppress_tokens=suppress_tokens,
        )

        print("Audio processing finished in: %s seconds\n" % round((time.time() - start_time2), 2))

        def re_form(s):
            s1 = s.rsplit('|', 1)[0]
            s2 = format(float(s.split("|")[-1]), '.2f')
            x = s1 + "| " + str(s2) + " audio seconds/s"
            return x

        def pbar_delayed():
            global timestamp_prev
            sleep(set_delay)
            pbar.update(round(timestamp_last - timestamp_prev))
            timestamp_prev = timestamp_last
            system("title " + re_form(capture.getvalue().splitlines()[-1]).replace("|", "^|").replace("<", "^<"))

        total_dur = round(info.duration)
        td_len = str(len(str(total_dur)))
        global timestamp_prev, timestamp_last
        timestamp_prev = 0
        timestamp_last = 0
        capture = io.StringIO()
        last_burst = 0.0
        set_delay = 0.1

        start_time3 = time.time()
        bar_format = "{percentage:3.0f}% | {n_fmt:>"+td_len+"}/{total_fmt} | {elapsed}<<{remaining} | {rate}"

        all_segments = []
        with tqdm(file=capture, total=total_dur, smoothing=0.00001, maxinterval=100000.0, bar_format=bar_format) as pbar:
            for segment in segments:
                progress=round(segment.start/total_dur*100,1)
                lefts=''
                if progress>1:
                    secondsLeft=(time.time()-start_time3)/progress*(100-progress);
                    if secondsLeft>60:
                        lefts=f"< {round(secondsLeft/60)} m"
                    else:
                        lefts=f"< {round(secondsLeft)} s"
                line = f"{progress}% {lefts} [{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}] {segment.text}"
                print(make_safe(line))
                segment_duration = str(segment.end - segment.start)
                all_segments.append(segment)

                timestamp_last = round(segment.end)
                time_now = time.time()

                if time_now - last_burst > set_delay:
                    last_burst = time_now
                    Thread(target=pbar_delayed, daemon=False).start()

        print("\nTranscription speed: %s audio seconds/s" % round(info.duration / ((time.time() - start_time3)), 2))
        print("Storing transcription")
        writer({"segments": all_segments}, audio_path, writer_args)
        print("\nOperation finished in: %s seconds" % int(round((time.time() - start_time))))
        sys.exit(0)
if __name__ == "__main__":
    cli()
