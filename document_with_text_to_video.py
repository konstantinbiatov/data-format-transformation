import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import fitz  # PyMuPDF


def run_command(command, input_text=None):
    """Run an external command and raise an error if it fails."""
    print("Running:", " ".join(map(str, command)))

    result = subprocess.run(
        command,
        input=input_text,
        text=True if input_text is not None else False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}"
        )

    return result


def check_program(program):
    if shutil.which(program) is None:
        raise FileNotFoundError(
            f"Required program was not found in PATH: {program}"
        )


def convert_to_pdf(input_file, output_dir):
    """
    Return a PDF path.
    If the input is already PDF, return it directly.
    If it is DOC/DOCX, convert it using LibreOffice.
    """
    input_file = Path(input_file)
    extension = input_file.suffix.lower()

    if extension == ".pdf":
        return input_file

    if extension not in [".doc", ".docx"]:
        raise ValueError(
            "Input file must have a .pdf, .doc, or .docx extension."
        )

    check_program("libreoffice")

    output_pdf = output_dir / f"{input_file.stem}.pdf"

    run_command([
        "libreoffice",
        "--headless",
        "--convert-to",
        "pdf",
        "--outdir",
        str(output_dir),
        str(input_file),
    ])

    converted_pdf = output_dir / f"{input_file.stem}.pdf"

    if not converted_pdf.exists():
        raise FileNotFoundError(
            f"LibreOffice did not create the expected PDF: {converted_pdf}"
        )

    return converted_pdf


def read_text_lines(text_file):
    """
    Read the text file.

    Empty lines are preserved because every line represents one page.
    """
    with open(text_file, "r", encoding="utf-8-sig") as file:
        content = file.read()

    # Normalize Windows and old Mac line endings.
    content = content.replace("\r\n", "\n").replace("\r", "\n")

    # split("\n") preserves empty lines between pages.
    lines = content.split("\n")

    # Remove only a possible final empty item caused by the final newline.
    if lines and lines[-1] == "":
        lines.pop()

    return lines


def render_pdf_pages(pdf_path, output_dir, dpi=150):
    """
    Render every PDF page as a PNG image.
    """
    document = fitz.open(pdf_path)
    image_files = []

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    for page_number, page in enumerate(document, start=1):
        print(f"Rendering PDF page {page_number}")

        pixmap = page.get_pixmap(
            matrix=matrix,
            alpha=False
        )

        image_path = output_dir / f"page_{page_number:05d}.png"
        pixmap.save(str(image_path))
        image_files.append(image_path)

    page_count = len(document)
    document.close()

    return image_files, page_count


def synthesize_with_piper(
    text,
    output_wav,
    piper_executable,
    piper_model,
    speaker=None,
):
    """
    Synthesize one text line using Piper.

    Piper receives text through standard input.
    """
    command = [
        piper_executable,
        "--model",
        str(piper_model),
        "--output_file",
        str(output_wav),
    ]

    if speaker is not None:
        command.extend(["--speaker", str(speaker)])

    run_command(command, input_text=text + "\n")


def create_page_video(
    image_path,
    audio_path,
    output_path,
    fps=30,
    width=1920,
    height=1080,
):
    """
    Create an MP4 showing one page for exactly the duration of its audio.
    The page is scaled to fit inside the video frame.
    """
    command = [
        "ffmpeg",
        "-y",

        # Image input.
        "-loop",
        "1",
        "-framerate",
        str(fps),
        "-i",
        str(image_path),

        # Audio input.
        "-i",
        str(audio_path),

        # Scale page to fit while preserving aspect ratio.
        "-vf",
        (
            f"scale={width}:{height}:"
            "force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:"
            "color=white"
        ),

        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",

        "-c:a",
        "aac",
        "-b:a",
        "192k",

        "-pix_fmt",
        "yuv420p",

        # Stop when the speech ends.
        "-shortest",

        str(output_path),
    ]

    run_command(command)


def concatenate_videos(video_files, output_file, work_dir):
    """
    Concatenate page videos using an FFmpeg concat file.
    """
    concat_file = work_dir / "concat.txt"

    with open(concat_file, "w", encoding="utf-8") as file:
        for video_file in video_files:
            # FFmpeg concat files use single-quoted paths.
            safe_path = str(video_file.resolve()).replace("'", "'\\''")
            file.write(f"file '{safe_path}'\n")

    command = [
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
        str(output_file),
    ]

    run_command(command)


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDF/DOC/DOCX pages and text lines into a narrated MP4."
    )

    parser.add_argument(
        "document",
        help="Input PDF, DOC, or DOCX file"
    )

    parser.add_argument(
        "text_file",
        help="Text file containing exactly one line per page"
    )

    parser.add_argument(
        "output",
        help="Output MP4 file"
    )

    parser.add_argument(
        "--piper",
        default="piper",
        help="Piper executable path, default: piper"
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to Piper .onnx model"
    )

    parser.add_argument(
        "--speaker",
        type=int,
        default=None,
        help="Speaker number for multi-speaker Piper models"
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="PDF rendering resolution, default: 150"
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Video frame rate, default: 30"
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1920,
        help="Video width, default: 1920"
    )

    parser.add_argument(
        "--height",
        type=int,
        default=1080,
        help="Video height, default: 1080"
    )

    args = parser.parse_args()

    document_path = Path(args.document).resolve()
    text_path = Path(args.text_file).resolve()
    output_path = Path(args.output).resolve()
    model_path = Path(args.model).resolve()

    if not document_path.exists():
        raise FileNotFoundError(document_path)

    if not text_path.exists():
        raise FileNotFoundError(text_path)

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    check_program(args.piper)
    check_program("ffmpeg")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="pdf_to_mp4_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)

        pdf_dir = temp_dir / "pdf"
        pages_dir = temp_dir / "pages"
        audio_dir = temp_dir / "audio"
        video_dir = temp_dir / "videos"

        pdf_dir.mkdir()
        pages_dir.mkdir()
        audio_dir.mkdir()
        video_dir.mkdir()

        # Convert DOC/DOCX to PDF if necessary.
        pdf_path = convert_to_pdf(document_path, pdf_dir)

        # Read one text line per page.
        lines = read_text_lines(text_path)

        # Render PDF pages.
        image_files, page_count = render_pdf_pages(
            pdf_path,
            pages_dir,
            dpi=args.dpi
        )

        if len(lines) != page_count:
            raise ValueError(
                f"Page count and line count do not match.\n"
                f"PDF pages: {page_count}\n"
                f"Text lines: {len(lines)}"
            )

        page_videos = []

        for page_number, (image_file, line) in enumerate(
            zip(image_files, lines),
            start=1
        ):
            print(f"\nProcessing page {page_number}/{page_count}")

            audio_file = audio_dir / f"audio_{page_number:05d}.wav"
            video_file = video_dir / f"video_{page_number:05d}.mp4"

            # Piper cannot synthesize an empty line meaningfully.
            # Create a very short silent audio file for empty lines.
            if line.strip():
                synthesize_with_piper(
                    text=line,
                    output_wav=audio_file,
                    piper_executable=args.piper,
                    piper_model=model_path,
                    speaker=args.speaker,
                )
            else:
                run_command([
                    "ffmpeg",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "anullsrc=channel_layout=stereo:sample_rate=22050",
                    "-t",
                    "0.5",
                    "-c:a",
                    "pcm_s16le",
                    str(audio_file),
                ])

            create_page_video(
                image_path=image_file,
                audio_path=audio_file,
                output_path=video_file,
                fps=args.fps,
                width=args.width,
                height=args.height,
            )

            page_videos.append(video_file)

        concatenate_videos(
            video_files=page_videos,
            output_file=output_path,
            work_dir=temp_dir,
        )

    print(f"\nFinished successfully:")
    print(output_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
