import json
import os.path
import typing

from faster_whisper.utils import format_timestamp
from faster_whisper.version import __version__

from os import system
from time import sleep

class ResultWriter:
    extension: str

    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def __call__(self, result: dict, audio_path: str):
        audio_basename = os.path.basename(audio_path)
        audio_basename = os.path.splitext(audio_basename)[0]
        output_path = os.path.join(
            self.output_dir, audio_basename + "." + self.extension
        )

        with open(output_path, "w", encoding="utf-8") as f:
            self.write_result(result, file=f)

    def write_result(self, result: dict, file: typing.TextIO):
        raise NotImplementedError


class WriteTXT(ResultWriter):
    extension: str = "txt"

    def write_result(self, result: dict, file: typing.TextIO):
        for segment in result["segments"]:
            print(f"[{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}]  {segment.text.strip()}", file=file, flush=True)


class SubtitlesWriter(ResultWriter):
    always_include_hours: bool
    decimal_marker: str

    def iterate_result(self, result: dict):
        for segment in result["segments"]:
            segment_start = self.format_timestamp(segment.start)
            segment_end = self.format_timestamp(segment.end)
            segment_text = segment.text.strip().replace("-->", "->")

            if word_timings := segment.words:
                all_words = [timing.word for timing in word_timings]
                all_words[0] = all_words[0].strip()  # remove the leading space, if any
                last = segment_start
                for i, this_word in enumerate(word_timings):
                    start = self.format_timestamp(this_word.start)
                    end = self.format_timestamp(this_word.end)
                    if last != start:
                        yield last, start, segment_text

                    yield start, end, "".join(
                        [
                            f"<u>{word}</u>" if j == i else word
                            for j, word in enumerate(all_words)
                        ]
                    )
                    last = end

                if last != segment_end:
                    yield last, segment_end, segment_text
            else:
                yield segment_start, segment_end, segment_text

    def format_timestamp(self, seconds: float):
        return format_timestamp(
            seconds=seconds,
            always_include_hours=self.always_include_hours,
            decimal_marker=self.decimal_marker,
        )


class WriteVTT(SubtitlesWriter):
    extension: str = "vtt"
    always_include_hours: bool = False
    decimal_marker: str = "."

    def write_result(self, result: dict, file: typing.TextIO):
        print("WEBVTT\n", file=file)
        for start, end, text in self.iterate_result(result):
            print(f"{start} --> {end}\n{text}\n", file=file, flush=True)


class WriteSRT(SubtitlesWriter):
    extension: str = "srt"
    always_include_hours: bool = True
    decimal_marker: str = ","

    def write_result(self, result: dict, file: typing.TextIO):
        for i, (start, end, text) in enumerate(self.iterate_result(result), start=1):
            print(f"{i}\n{start} --> {end}\n{text}\n", file=file, flush=True)


class WriteTSV(ResultWriter):
    """
    Write a transcript to a file in TSV (tab-separated values) format containing lines like:
    <start time in integer milliseconds>\t<end time in integer milliseconds>\t<transcript text>

    Using integer milliseconds as start and end times means there's no chance of interference from
    an environment setting a language encoding that causes the decimal in a floating point number
    to appear as a comma; also is faster and more efficient to parse & store, e.g., in C++.
    """

    extension: str = "tsv"

    def write_result(self, result: dict, file: typing.TextIO):
        print("start", "end", "text", sep="\t", file=file)
        for segment in result["segments"]:
            print(round(1000 * segment.start), file=file, end="\t")
            print(round(1000 * segment.end), file=file, end="\t")
            print(segment.text.strip().replace("\t", " "), file=file, flush=True)


class WriteJSON(ResultWriter):
    extension: str = "json"

    def write_result(self, result: dict, file: typing.TextIO):
        json.dump(result, file)


def get_writer(output_format: str, output_dir: str) -> typing.Callable[[dict, typing.TextIO], None]:
    writers = {
        "txt": WriteTXT,
        "vtt": WriteVTT,
        "srt": WriteSRT,
        "tsv": WriteTSV,
        "json": WriteJSON,
    }

    if output_format == "all":
        all_writers = [writer(output_dir) for writer in writers.values()]

        def write_all(result: dict, file: typing.TextIO):
            for writer in all_writers:
                writer(result, file)

        return write_all

    return writers[output_format](output_dir)
