#!/usr/bin/env python3

import argparse
import shutil
import subprocess
import sys
import wave
from pathlib import Path

from pypdf import PdfReader
from pdf2image import convert_from_path


def run_command(command):
    """Run a command and stop if it fails."""
    print("Running:", " ".join(map(str, command)))

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}"
        )

    return result


def check_program(program):
    if shutil.which(program) is None:
        raise RuntimeError(
            f"Required program not found: {program}"
        )


def convert_to_pdf(input_file, work_dir):
    """
    Convert PDF/DOC/DOCX/PPT to PDF.
    """
    input_file = Path(input_file)

    if input_file.suffix.lower() == ".pdf":
        pdf_file = work_dir / input_file.name
        shutil.copy2(input_file, pdf_file)
        return pdf_file

    office_extensions = {".doc", ".docx", ".ppt"}

    if input_file.suffix.lower() not in office_extensions:
        raise ValueError(
            "Input must have one of these extensions: "
            ".pdf, .doc, .docx, .ppt"
        )

    office_output_dir = work_dir / "office_pdf"
    office_output_dir.mkdir(exist_ok=True)

    run_command([
        "libreoffice",
        "--headless",
        "--convert-to",
        "pdf",
        "--outdir",
        str(office_output_dir),
        str(input_file)
    ])

    generated_pdf = office_output_dir / f"{input_file.stem}.pdf"

    if not generated_pdf.exists():
        raise RuntimeError(
            f"LibreOffice did not create expected PDF: {generated_pdf}"
        )

    return generated_pdf


def extract_page_text(pdf_file, text_dir):
    """
    Extract text from every PDF page.
    Returns a list of text file paths.
    """
    reader = PdfReader(str(pdf_file))
    text_files = []

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            print(
                f"Warning: could not extract text from page "
                f"{page_number}: {exc}"
            )
            text = ""

        text_file = text_dir / f"page_{page_number:04d}.txt"
        text_file.write_text(text.strip(), encoding="utf-8")
        text_files.append(text_file)

    return text_files


def render_pages(pdf_file, image_dir, dpi):
    """
    Render each PDF page as a PNG image.
    """
    print("Rendering pages...")

    images = convert_from_path(
        str(pdf_file),
        dpi=dpi,
        fmt="png"
    )

    image_files = []

    for page_number, image in enumerate(images, start=1):
        image_file = image_dir / f"page_{page_number:04d}.png"
        image.save(image_file, "PNG")
        image_files.append(image_file)

    return image_files


def wav_duration(wav_file):
    """
    Return WAV duration in seconds.
    """
    with wave.open(str(wav_file), "rb") as wav:
        frames = wav.getnframes()
        rate = wav.getframerate()

    return frames / float(rate)


def generate_piper_audio(text_file, wav_file, piper_model):
    """
    Generate WAV audio using Piper.
    """
    text = text_file.read_text(encoding="utf-8").strip()

    if not text:
        # Generate a short silent WAV if the page has no extractable text.
        run_command([
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=22050:cl=mono",
            "-t",
            "0.5",
            str(wav_file)
        ])
        return

    with text_file.open("r", encoding="utf-8") as source:
        command = [
            "piper",
            "--model",
            str(piper_model),
            "--output_file",
            str(wav_file)
        ]

        result = subprocess.run(
            command,
            stdin=source,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError("Piper TTS failed")

    if not wav_file.exists():
        raise RuntimeError(f"Piper did not create: {wav_file}")


def create_page_video(image_file, wav_file, output_file, delay):
    """
    Create a video segment for one page.

    The image is displayed for:
        audio duration + delay
    """

    audio_length = wav_duration(wav_file)
    total_length = audio_length + delay

    run_command([
        "ffmpeg",
        "-y",

        # Loop the page image for the complete segment.
        "-loop",
        "1",
        "-i",
        str(image_file),

        # Page audio.
        "-i",
        str(wav_file),

        "-t",
        str(total_length),

        "-vf",
        (
            "scale=trunc(iw/2)*2:"
            "trunc(ih/2)*2,"
            "format=yuv420p"
        ),

        "-r",
        "25",

        # Make sure audio ends cleanly before the silent delay.
        "-af",
        f"apad=pad_dur={delay}",

        "-c:v",
        "libx264",

        "-preset",
        "medium",

        "-c:a",
        "aac",

        "-b:a",
        "192k",

        "-shortest",

        str(output_file)
    ])


def create_concat_file(segment_files, concat_file):
    """
    Create an FFmpeg concat input file.
    """
    with concat_file.open("w", encoding="utf-8") as file:
        for segment in segment_files:
            # FFmpeg concat files use single-quoted paths.
            safe_path = str(segment.resolve()).replace("'", "'\\''")
            file.write(f"file '{safe_path}'\n")


def join_segments(segment_files, output_mp4, concat_file):
    """
    Join all page video segments into one MP4.
    """
    create_concat_file(segment_files, concat_file)

    run_command([
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        str(output_mp4)
    ])


def process_document(
    input_file,
    output_dir,
    piper_model,
    delay,
    dpi
):
    input_file = Path(input_file).resolve()
    output_dir = Path(output_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    work_dir = output_dir / "work"
    text_dir = output_dir / "text"
    image_dir = output_dir / "pages"
    audio_dir = output_dir / "audio"
    segment_dir = output_dir / "segments"

    for directory in [
        work_dir,
        text_dir,
        image_dir,
        audio_dir,
        segment_dir
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    print("Converting input document to PDF...")
    pdf_file = convert_to_pdf(input_file, work_dir)

    print("Extracting text...")
    text_files = extract_page_text(pdf_file, text_dir)

    print("Rendering pages...")
    image_files = render_pages(pdf_file, image_dir, dpi)

    if len(text_files) != len(image_files):
        raise RuntimeError(
            "Number of text pages does not match number of images"
        )

    segment_files = []

    for index, (text_file, image_file) in enumerate(
        zip(text_files, image_files),
        start=1
    ):
        print(f"Processing page {index}/{len(image_files)}")

        wav_file = audio_dir / f"page_{index:04d}.wav"

        generate_piper_audio(
            text_file=text_file,
            wav_file=wav_file,
            piper_model=piper_model
        )

        segment_file = segment_dir / f"page_{index:04d}.mp4"

        create_page_video(
            image_file=image_file,
            wav_file=wav_file,
            output_file=segment_file,
            delay=delay
        )

        segment_files.append(segment_file)

    output_mp4 = output_dir / f"{input_file.stem}.mp4"
    concat_file = output_dir / "concat.txt"

    print("Joining all pages...")
    join_segments(
        segment_files=segment_files,
        output_mp4=output_mp4,
        concat_file=concat_file
    )

    print()
    print("Finished.")
    print(f"Video: {output_mp4}")
    print(f"Text files: {text_dir}")
    print(f"Audio files: {audio_dir}")
    print(f"Page images: {image_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDF/DOC/DOCX/PPT to one Piper-TTS MP4 video."
    )

    parser.add_argument(
        "input_file",
        help="Input document: PDF, DOC, DOCX, or PPT"
    )

    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory"
    )

    parser.add_argument(
        "--piper-model",
        required=True,
        help="Path to Piper .onnx voice model"
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Silence/display delay between pages in seconds"
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Page rendering resolution"
    )

    args = parser.parse_args()

    check_program("ffmpeg")
    check_program("libreoffice")
    check_program("piper")

    piper_model = Path(args.piper_model)

    if not piper_model.exists():
        raise FileNotFoundError(
            f"Piper model not found: {piper_model}"
        )

    process_document(
        input_file=args.input_file,
        output_dir=args.output_dir,
        piper_model=piper_model,
        delay=args.delay,
        dpi=args.dpi
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
