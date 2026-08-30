import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from vosk import Model, KaldiRecognizer


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov"}

# Use a Unicode font if available on your system.
FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

def run_command(command):
    """Run a command and raise an error if it fails."""
    print("Running:", " ".join(map(str, command)))

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"Command failed: {' '.join(map(str, command))}")

    return result


def get_font(size):
    for font_path in FONT_PATHS:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size=size)

    return ImageFont.load_default()


def format_timestamp(seconds):
    milliseconds = int((seconds - int(seconds)) * 1000)
    total_seconds = int(seconds)

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


def wrap_text(draw, text, font, max_width):
    """
    Wrap text so that each line fits within max_width.
    """
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        test_line = word if not current_line else current_line + " " + word
        bbox = draw.textbbox((0, 0), test_line, font=font)
        line_width = bbox[2] - bbox[0]

        if line_width <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    return lines


# ------------------------------------------------------------
# Video processing
# ------------------------------------------------------------

def get_video_info(video_path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        raise RuntimeError("Could not determine video FPS.")

    duration = frame_count / fps

    cap.release()

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": duration,
    }


def detect_distinguishing_frames(
    video_path,
    distance_threshold=12.0,
    resize_width=320,
    minimum_gap_seconds=0.25
):
    """
    Detect frames whose grayscale image differs significantly from
    the previously selected frame.

    distance_threshold:
        Mean absolute grayscale pixel difference.
        Higher value means fewer detected frames.

    minimum_gap_seconds:
        Prevents many nearly identical frames from being selected.
    """

    info = get_video_info(video_path)
    fps = info["fps"]

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    selected_frames = []
    previous_gray = None
    last_selected_time = -999999

    frame_index = 0

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        timestamp = frame_index / fps

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if resize_width and gray.shape[1] > resize_width:
            new_height = int(gray.shape[0] * resize_width / gray.shape[1])
            gray = cv2.resize(gray, (resize_width, new_height))

        if previous_gray is None:
            selected_frames.append({
                "index": frame_index,
                "timestamp": timestamp,
                "frame": frame.copy(),
            })

            previous_gray = gray
            last_selected_time = timestamp
            frame_index += 1
            continue

        distance = np.mean(
            cv2.absdiff(gray, previous_gray).astype(np.float32)
        )

        enough_time_passed = (
            timestamp - last_selected_time >= minimum_gap_seconds
        )

        if distance >= distance_threshold and enough_time_passed:
            selected_frames.append({
                "index": frame_index,
                "timestamp": timestamp,
                "frame": frame.copy(),
                "distance": float(distance),
            })

            last_selected_time = timestamp
            previous_gray = gray

        frame_index += 1

    cap.release()

    if not selected_frames:
        raise RuntimeError("No distinguishing frames were detected.")

    # Add the final video timestamp as a sentinel boundary.
    # This guarantees one audio chunk for every selected frame.
    final_timestamp = info["duration"]

    selected_frames.append({
        "index": info["frame_count"] - 1,
        "timestamp": final_timestamp,
        "frame": None,
        "is_end_boundary": True,
    })

    return selected_frames, info


# ------------------------------------------------------------
# Audio extraction and Vosk transcription
# ------------------------------------------------------------

def extract_audio_chunk(video_path, output_wav, start_time, end_time):
    """
    Extract mono, 16 kHz, 16-bit PCM WAV audio.
    This is the format expected by Vosk.
    """

    duration = max(0.01, end_time - start_time)

    command = [
        "ffmpeg",
        "-y",
        "-ss", f"{start_time:.3f}",
        "-i", str(video_path),
        "-t", f"{duration:.3f}",
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-sample_fmt", "s16",
        "-f", "wav",
        str(output_wav),
    ]

    run_command(command)


def transcribe_wav(wav_path, vosk_model):
    """
    Transcribe a WAV file with Vosk.
    """

    import wave

    with wave.open(str(wav_path), "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()

        if channels != 1:
            raise ValueError("Audio must be mono.")

        if sample_width != 2:
            raise ValueError("Audio must be 16-bit PCM.")

        recognizer = KaldiRecognizer(vosk_model, sample_rate)
        recognizer.SetWords(True)

        while True:
            data = wf.readframes(4000)

            if not data:
                break

            recognizer.AcceptWaveform(data)

        final_result = json.loads(recognizer.FinalResult())
        return final_result.get("text", "").strip()


# ------------------------------------------------------------
# Frame extraction and image creation
# ------------------------------------------------------------

def get_frame_at_time(video_path, timestamp):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)

    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError(f"Could not read frame at {timestamp} seconds.")

    return frame


def create_combined_jpg(
    frame_bgr,
    transcript,
    start_time,
    end_time,
    output_path,
    page_width=1800,
    margin=60
):
    """
    Create a white image with the video frame on the left
    and transcript on the right.
    """

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_image = Image.fromarray(frame_rgb)

    # Fit frame into the left half of the output image.
    left_width = int(page_width * 0.52)
    right_width = page_width - left_width - margin * 3

    frame_scale = min(
        left_width / frame_image.width,
        page_width / frame_image.height
    )

    resized_frame = frame_image.resize(
        (
            int(frame_image.width * frame_scale),
            int(frame_image.height * frame_scale)
        ),
        Image.Resampling.LANCZOS
    )

    image_height = max(resized_frame.height + margin * 2, 900)

    canvas = Image.new(
        "RGB",
        (page_width, image_height),
        "white"
    )

    draw = ImageDraw.Draw(canvas)

    # Place frame on the left.
    frame_x = margin
    frame_y = (image_height - resized_frame.height) // 2

    canvas.paste(resized_frame, (frame_x, frame_y))

    # Text area on the right.
    text_x = left_width + margin * 2
    text_y = margin

    title_font = get_font(36)
    body_font = get_font(32)
    small_font = get_font(25)

    draw.text(
        (text_x, text_y),
        f"{format_timestamp(start_time)} - "
        f"{format_timestamp(end_time)}",
        fill="black",
        font=title_font
    )

    text_y += 70

    draw.text(
        (text_x, text_y),
        "Transcript:",
        fill="black",
        font=title_font
    )

    text_y += 70

    if not transcript:
        transcript = "[No speech detected]"

    lines = wrap_text(
        draw,
        transcript,
        body_font,
        right_width
    )

    line_height = 48

    for line in lines:
        if text_y + line_height > image_height - margin:
            # Add a second page-like continuation inside the same image.
            break

        draw.text(
            (text_x, text_y),
            line,
            fill="black",
            font=body_font
        )

        text_y += line_height

    draw.text(
        (text_x, image_height - margin - 35),
        "Decoded with Vosk",
        fill="gray",
        font=small_font
    )

    canvas.save(
        output_path,
        "JPEG",
        quality=95,
        subsampling=0
    )


# ------------------------------------------------------------
# PDF creation
# ------------------------------------------------------------

def combine_jpgs_into_pdf(jpg_files, output_pdf):
    if not jpg_files:
        raise RuntimeError("No JPG files were created.")

    images = []

    for jpg_file in jpg_files:
        image = Image.open(jpg_file).convert("RGB")
        images.append(image)

    first_image = images[0]
    remaining_images = images[1:]

    first_image.save(
        output_pdf,
        "PDF",
        resolution=150.0,
        save_all=True,
        append_images=remaining_images
    )


# ------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------

def process_video(
    video_path,
    vosk_model_path,
    output_dir,
    distance_threshold=12.0,
    minimum_gap_seconds=0.25
):
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    audio_dir = output_dir / "audio_chunks"
    jpg_dir = output_dir / "jpg_pages"

    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(exist_ok=True)
    jpg_dir.mkdir(exist_ok=True)

    print("Loading Vosk model...")
    vosk_model = Model(str(vosk_model_path))

    print("Detecting distinguishing frames...")

    selected_frames, video_info = detect_distinguishing_frames(
        video_path=video_path,
        distance_threshold=distance_threshold,
        minimum_gap_seconds=minimum_gap_seconds
    )

    # The final element is only a boundary, not a real content frame.
    content_frames = selected_frames[:-1]
    boundaries = [item["timestamp"] for item in selected_frames]

    print(f"Detected frames: {len(content_frames)}")
    print(f"Audio chunks: {len(boundaries) - 1}")
    print(f"Video duration: {video_info['duration']:.2f} seconds")

    jpg_files = []

    for i, frame_info in enumerate(content_frames):
        start_time = boundaries[i]
        end_time = boundaries[i + 1]

        if end_time <= start_time:
            continue

        print(
            f"\nProcessing chunk {i + 1}/{len(content_frames)}: "
            f"{start_time:.2f} - {end_time:.2f} seconds"
        )

        audio_file = audio_dir / f"chunk_{i:05d}.wav"
        jpg_file = jpg_dir / f"page_{i:05d}.jpg"

        extract_audio_chunk(
            video_path=video_path,
            output_wav=audio_file,
            start_time=start_time,
            end_time=end_time
        )

        transcript = transcribe_wav(
            wav_path=audio_file,
            vosk_model=vosk_model
        )

        print("Transcript:", transcript)

        frame = frame_info.get("frame")

        if frame is None:
            frame = get_frame_at_time(video_path, start_time)

        create_combined_jpg(
            frame_bgr=frame,
            transcript=transcript,
            start_time=start_time,
            end_time=end_time,
            output_path=jpg_file
        )

        jpg_files.append(jpg_file)

    output_pdf = output_dir / "combined_transcript.pdf"

    print("\nCreating PDF...")
    combine_jpgs_into_pdf(jpg_files, output_pdf)

    print(f"\nFinished: {output_pdf}")


# ------------------------------------------------------------
# Command-line interface
# ------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect video frames, transcribe matching audio chunks, "
                    "and create a combined PDF."
    )

    parser.add_argument(
        "video",
        help="Input MP4 video file"
    )

    parser.add_argument(
        "--vosk-model",
        required=True,
        help="Path to the downloaded Vosk model directory"
    )

    parser.add_argument(
        "--output",
        default="output",
        help="Output directory"
    )

    parser.add_argument(
        "--distance",
        type=float,
        default=12.0,
        help=(
            "Grayscale frame distance threshold. "
            "Lower values detect more frames. Default: 12.0"
        )
    )

    parser.add_argument(
        "--min-gap",
        type=float,
        default=0.25,
        help=(
            "Minimum time between detected frames in seconds. "
            "Default: 0.25"
        )
    )

    args = parser.parse_args()

    process_video(
        video_path=args.video,
        vosk_model_path=args.vosk_model,
        output_dir=args.output,
        distance_threshold=args.distance,
        minimum_gap_seconds=args.min_gap
    )
