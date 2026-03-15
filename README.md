# mathdoc-parser

`mathdoc-parser` is a small tool for converting mathematical documents into structured JSON through a three-stage pipeline:

`PDF -> Markdown -> LaTeX -> JSON`

The repository supports two modes:

- `book`: tuned for textbook-style layouts
- `paper`: tuned for academic papers

## Layout

- `main.py`: pipeline runner with resume support
- `src/book/`: book-specific OCR, Markdown-to-LaTeX, and LaTeX-to-JSON stages
- `src/paper/`: paper-specific OCR, Markdown-to-LaTeX, and LaTeX-to-JSON stages
- `settings.json`: root settings for the main runner
- `src/book/settings.json`: stage settings for the book pipeline
- `src/paper/settings.json`: stage settings for the paper pipeline
- `config.example.json`: template for API credentials and model names

## Configuration

Create a local `config.json` from `config.example.json` and fill in:

- `api_key`
- `base_url`
- `model`

`config.json` is ignored by Git.

Runner settings live in the root `settings.json`.

Stage settings live in:

- `src/book/settings.json`
- `src/paper/settings.json`

## Usage

Put input PDFs under one of these directories:

- `input_pdfs/book/`
- `input_pdfs/paper/`

Run the book pipeline:

```bash
python main.py --mode book
```

Run the paper pipeline:

```bash
python main.py --mode paper
```

Outputs are written to:

- `work/...` for intermediate Markdown and LaTeX files
- `output_json/...` for final JSON files

The directory structure under `work/` and `output_json/` mirrors the relative path under `input_pdfs/`.

## Notes

- `main.py` supports resume mode and atomic stage outputs.
- Book and paper files stay separated by the input directory structure.
- The repository ignores local inputs, outputs, working directories, caches, and private config files.

## License

Released under the Apache 2.0 license. See LICENSE for details.
