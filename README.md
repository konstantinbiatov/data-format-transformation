

How it works

python3 -m venv myenv
source myenv/bin/activate

pip3 install -r requirements.txt




#Download VOSK ASR models

https://alphacephei.com/vosk/models
Vosk provides models for many languages, including English, Spanish, French, German, Chinese, and others.


Common choices include:

- **Small models** — usually around 40–100 MB; faster and easier to run locally.
- **Large models** — often several hundred MB or more; generally more accurate but require more memory and storage.

For English, choose an English model such as an `en-us` model. Download the `.zip` archive.

**Extract the model**

Linux or macOS:

```bash
unzip vosk-model-small-en-us-*.zip

For example:

vosk-model-small-en-us-0.15

Run it like this:

```bash
python video_to_pdf.py input.mp4 \
    --vosk-model /path/to/vosk-model-small-en-us-0.15 \
    --output result \
    --distance 12
```

The output structure will be:

```text
result/
├── audio_chunks/
│   ├── chunk_00000.wav
│   ├── chunk_00001.wav
│   └── ...
├── jpg_pages/
│   ├── page_00000.jpg
│   ├── page_00001.jpg
│   └── ...
└── combined_transcript.pdf
```

The `--distance` parameter controls frame detection:

- `5–8`: detects small visual changes and creates many chunks.
- `10–15`: reasonable starting range.
- `20–30`: detects only major scene changes and creates fewer chunks.

The final video timestamp is added as a boundary, so each detected frame corresponds to exactly one audio chunk.

----------------------------------
# Download the TTS models

For an English document:

# Download the TTS model file, english - *.onnx
wget "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx?download=true" \
     -O en_US-amy-medium.onnx


# Download the TTS model config, english - *.onnx.json
wget "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx.json?download=true" \
     -O en_US-amy-medium.onnx.json


```bash
python document_to_video.py presentation.ppt \
  --piper-model voices/en_US-lessac-medium.onnx \
  --delay 3 \
  --output-dir result
```

For a Word document:

```bash
python document_to_video.py document.docx \
  --piper-model voices/en_US-lessac-medium.onnx \
  --delay 2.5
```

For a PDF:

```bash
python document_to_video.py book.pdf \
  --piper-model voices/en_US-lessac-medium.onnx \
  --delay 1
```

The resulting directory will contain:

```text
output/
├── document.mp4
├── audio/
│   ├── page_0001.wav
│   ├── page_0002.wav
│   └── ...
├── pages/
│   ├── page_0001.png
│   ├── page_0002.png
│   └── ...
└── text/
    ├── page_0001.txt
    ├── page_0002.txt
    └── ...
```

The `--piper-model` parameter controls the language and voice. For scanned PDFs containing images instead of selectable text, OCR must be added before Piper can read the pages.

---------------------
#DOCUMENT TO QRCODE VIDEO

Example usage:

```bash
python document_to_qr_video.py presentation.pptx \
    --output result \
    --delay 2
```

For a PDF:

```bash
python document_to_qr_video.py document.pdf \
    --output result \
    --dpi 200 \
    --chunk-size 1600 \
    --delay 1.5
```

The generated directory will look like this:

```text
result/
├── pages/
│   ├── page_00001.png
│   ├── page_00002.png
│   └── ...
├── qrcodes/
│   ├── page_00001_part_00001.png
│   ├── page_00001_part_00002.png
│   ├── page_00002_part_00001.png
│   └── ...
└── document.mp4
```

The QR data contains metadata identifying:

- Document ID
- Page number
- Total page count
- Chunk number
- Total chunks for the page
- SHA-256 checksum of each chunk

To make the QR codes easier to scan, reduce the payload size:

```bash
python document_to_qr_video.py file.pdf \
    --chunk-size 1000 \
    --box-size 15 \
    --delay 2
```

A smaller `--chunk-size` produces more QR frames but gives better scanning reliability.


###################################

## Usage

For a PDF:

```bash
python pdf_with_text_to_video.py \
    document.pdf \
    text.txt \
    result.mp4 \
    --model en_US-lessac-medium.onnx
```

For a DOCX:

```bash
python pdf_with_text_to_video.py \
    document.docx \
    text.txt \
    result.mp4 \
    --model en_US-lessac-medium.onnx
```

If Piper is not in the system `PATH`, specify its path:

```bash
python pdf_with_text_to_video.py \
    document.pdf \
    text.txt \
    result.mp4 \
    --piper /path/to/piper \
    --model /path/to/en_US-lessac-medium.onnx
```

The text file must look like this:

```text
This is the narration for page one.
This is the narration for page two.
This is the narration for page three.
```

The first line is synchronized with page 1, the second line with page 2, and so on. Each page remains visible for the entire duration of its synthesized speech.
