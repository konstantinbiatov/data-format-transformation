import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import fitz  # PyMuPDF
import numpy as np
import pytesseract
import qrcode
from PIL import Image
from qrcode.constants import ERROR_CORRECT_L
from qrcode.exceptions import DataOverflowError


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
}


def check_program(program_name):
    """
    Check whether an external executable is available.
    """
    if shutil.which(program_name) is None:
        raise RuntimeError(
            f"Required program was not found: {program_name}\n"
            f"Please install it and make sure it is available in PATH."
        )


def convert_office_to_pdf(input_file, output_dir):
    """
    Convert DOC/DOCX/PPT/PPTX to PDF using LibreOffice.
    """
    check_program("libreoffice")

    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        "libreoffice",
        "--headless",
        "--convert-to",
        "pdf",
        "--outdir",
        str(output_dir),
        str(input_file),
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "LibreOffice conversion failed:\n"
            + result.stderr
        )

    converted_pdf = output_dir / f"{input_file.stem}.pdf"

    if not converted_pdf.exists():
        # LibreOffice may generate a slightly different filename.
        pdf_files = list(output_dir.glob("*.pdf"))

        if not pdf_files:
            raise RuntimeError(
                f"Could not find converted PDF for {input_file}"
            )

        converted_pdf = pdf_files[0]

    return converted_pdf


def prepare_pdf(input_file, work_dir):
    """
    Return a PDF path. Office files are first converted to PDF.
    """
    extension = input_file.suffix.lower()

    if extension == ".pdf":
        return input_file

    if extension in {".doc", ".docx", ".ppt", ".pptx"}:
        conversion_dir = work_dir / "converted"
        return convert_office_to_pdf(input_file, conversion_dir)

    raise ValueError(
        f"Unsupported file format: {input_file.suffix}"
    )


def render_page(page, dpi=200):
    """
    Render a PDF page as a PIL image.
    """
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    pixmap = page.get_pixmap(
        matrix=matrix,
        alpha=False,
    )

    image = Image.frombytes(
        "RGB",
        [pixmap.width, pixmap.height],
        pixmap.samples,
    )

    return image


def normalize_text(text):
    """
    Clean extracted/OCR text while preserving line breaks.
    """
    lines = []

    for line in text.splitlines():
        line = " ".join(line.split())

        if line:
            lines.append(line)

    return "\n".join(lines).strip()


def extract_page_text(page, page_image, enable_ocr=True):
    """
    Extract normal PDF text and optionally OCR the rendered page.
    OCR is useful for text in images and scanned pages.
    """
    normal_text = page.get_text("text")

    ocr_text = ""

    if enable_ocr:
        # OCR the complete rendered page. This detects text in images,
        # scanned documents, screenshots, and ordinary page text.
        ocr_text = pytesseract.image_to_string(page_image)

    normal_text = normalize_text(normal_text)
    ocr_text = normalize_text(ocr_text)

    if normal_text and ocr_text:
        return normal_text + "\n" + ocr_text

    return normal_text or ocr_text


def split_utf8_text(text, max_bytes):
    """
    Split text by UTF-8 byte size.

    QR-code capacity is byte-based, not character-based.
    This method avoids splitting a Unicode character in the middle.
    """
    if not text:
        return [""]

    chunks = []
    current = []
    current_size = 0

    for character in text:
        character_size = len(character.encode("utf-8"))

        if current and current_size + character_size > max_bytes:
            chunks.append("".join(current))
            current = []
            current_size = 0

        current.append(character)
        current_size += character_size

    if current:
        chunks.append("".join(current))

    return chunks


def get_qr_capacity(text, error_correction=ERROR_CORRECT_L):
    """
    Find a QR version that can hold the text.
    Returns the QR version if it fits, otherwise None.
    """
    for version in range(1, 41):
        qr = qrcode.QRCode(
            version=version,
            error_correction=error_correction,
            box_size=10,
            border=4,
        )

        try:
            qr.add_data(text)
            qr.make(fit=False)
            return version
        except DataOverflowError:
            continue

    return None


def split_text_for_qr(text, max_chunk_bytes=2800):
    """
    Split text into chunks that fit safely in QR codes.

    A QR Code Version 40 has a larger theoretical capacity, but a
    smaller practical chunk size is easier to scan reliably.
    """
    if not text:
        return ["[EMPTY PAGE]"]

    # Start with a practical limit.
    chunks = split_utf8_text(text, max_chunk_bytes)

    final_chunks = []

    for chunk in chunks:
        if get_qr_capacity(chunk) is not None:
            final_chunks.append(chunk)
            continue

        # If a chunk is still too large, reduce it repeatedly.
        reduced_limit = max_chunk_bytes

        while get_qr_capacity(chunk) is None:
            reduced_limit = int(reduced_limit * 0.8)

            if reduced_limit < 100:
                raise RuntimeError(
                    "Could not split text into a QR-compatible size."
                )

            smaller_chunks = split_utf8_text(chunk, reduced_limit)

            if len(smaller_chunks) == 1:
                chunk = smaller_chunks[0]
            else:
                final_chunks.extend(smaller_chunks[:-1])
                chunk = smaller_chunks[-1]

        final_chunks.append(chunk)

    return final_chunks


def create_qr_png(text, output_file, page_number, part_number, total_parts):
    """
    Create a PNG QR code.
    """
    # Add metadata so chunks can be reordered when decoding.
    payload = (
        f"PAGE={page_number};"
        f"PART={part_number}/{total_parts}\n"
        f"{text}"
    )

    qr = qrcode.QRCode(
        error_correction=ERROR_CORRECT_L,
        box_size=12,
        border=5,
    )

    qr.add_data(payload)
    qr.make(fit=True)

    image = qr.make_image(
        fill_color="black",
        back_color="white",
    )

    image.save(output_file)


def extract_pages_and_create_qr_files(
    pdf_path,
    qr_dir,
    dpi=200,
    enable_ocr=True,
):
    """
    Process every PDF page and create QR PNG files.
    """
    qr_dir.mkdir(parents=True, exist_ok=True)

    document = fitz.open(pdf_path)
    qr_files = []

    if enable_ocr:
        try:
            check_program("tesseract")
        except RuntimeError as error:
            print(str(error), file=sys.stderr)
            print("Continuing with PDF text extraction only.", file=sys.stderr)
            enable_ocr = False

    for page_index in range(len(document)):
        page_number = page_index + 1
        page = document[page_index]

        page_image = render_page(page, dpi=dpi)
        page_text = extract_page_text(
            page,
            page_image,
            enable_ocr=enable_ocr,
        )

        if not page_text:
            page_text = "[NO TEXT FOUND ON PAGE]"

        chunks = split_text_for_qr(page_text)

        print(
            f"Page {page_number}: "
            f"{len(page_text.encode('utf-8'))} bytes, "
            f"{len(chunks)} QR part(s)"
        )

        for part_index, chunk in enumerate(chunks, start=1):
            filename = (
                f"page_{page_number:04d}_"
                f"part_{part_index:04d}.png"
            )

            output_file = qr_dir / filename

            create_qr_png(
                text=chunk,
                output_file=output_file,
                page_number=page_number,
                part_number=part_index,
                total_parts=len(chunks),
            )

            qr_files.append(output_file)

    document.close()

    return qr_files


def create_mp4_from_pngs(
    png_files,
    output_file,
    delay_seconds=2.0,
    fps=None,
):
    """
    Create an MP4 slideshow.

    delay_seconds means how long each QR image is displayed.
    """
    if not png_files:
        raise RuntimeError("No PNG files were generated.")

    if delay_seconds <= 0:
        raise ValueError("delay_seconds must be greater than zero.")

    if fps is None:
        fps = 1.0 / delay_seconds

    first_image = cv2.imread(str(png_files[0]))

    if first_image is None:
        raise RuntimeError(
            f"Could not read image: {png_files[0]}"
        )

    height, width = first_image.shape[:2]

    # MP4 encoders commonly require even dimensions.
    width -= width % 2
    height -= height % 2

    output_file.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_file),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not open MP4 writer: {output_file}"
        )

    try:
        for index, png_file in enumerate(png_files, start=1):
            image = cv2.imread(str(png_file))

            if image is None:
                print(
                    f"Skipping unreadable image: {png_file}",
                    file=sys.stderr,
                )
                continue

            image = cv2.resize(
                image,
                (width, height),
                interpolation=cv2.INTER_AREA,
            )

            writer.write(image)

            print(
                f"Added frame {index}/{len(png_files)}: "
                f"{png_file.name}"
            )

    finally:
        writer.release()

    print(f"MP4 created: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract text from document pages, convert it to QR-code "
            "PNGs, and create an MP4 slideshow."
        )
    )

    parser.add_argument(
        "input_file",
        type=Path,
        help="Input PDF, DOC, DOCX, PPT, or PPTX file",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Output directory",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds each QR frame is displayed",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Rendering resolution used for OCR",
    )

    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable OCR and extract only embedded document text",
    )

    args = parser.parse_args()

    input_file = args.input_file.resolve()

    if not input_file.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {input_file}"
        )

    if input_file.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            "Supported formats: PDF, DOC, DOCX, PPT, PPTX"
        )

    work_dir = args.output_dir / "work"
    qr_dir = args.output_dir / "qr_codes"
    mp4_file = args.output_dir / f"{input_file.stem}.mp4"

    work_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = prepare_pdf(
        input_file=input_file,
        work_dir=work_dir,
    )

    qr_files = extract_pages_and_create_qr_files(
        pdf_path=pdf_path,
        qr_dir=qr_dir,
        dpi=args.dpi,
        enable_ocr=not args.no_ocr,
    )

    create_mp4_from_pngs(
        png_files=qr_files,
        output_file=mp4_file,
        delay_seconds=args.delay,
    )

    print()
    print(f"QR PNG files: {qr_dir}")
    print(f"MP4 file:     {mp4_file}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
