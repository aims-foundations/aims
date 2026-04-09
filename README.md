# AI Measurement Science
### Sang Truong and Sanmi Koyejo

A Science of Knowing Where AI Thrives, Where It Breaks, and How to Respond

This book is available online at: [aimslab.stanford.edu/textbook/](https://aimslab.stanford.edu/textbook/)

## Prerequisites

Before building the book, ensure you have the following installed:

- **Quarto** >= 1.9.36 (CI pins 1.9.36)
- **Python** >= 3.8
- **R** >= 4.3.1

## Installation

### 1. Install Quarto

- **macOS (Homebrew):** `brew install quarto`
- **macOS/Windows:** Download from https://quarto.org/docs/get-started/

### 2. Install Python

- **macOS:** `brew install python`
- **Windows:** Download from https://www.python.org/downloads/

### 3. Install R

- **macOS:** `brew install r`
- **Windows:** Download from https://cran.r-project.org/

### 4. Clone the repository

```bash
git clone https://github.com/aims-foundations/aims
cd aims
```

### 5. Install Python build dependencies

```bash
pip install -r requirements-build.txt
```

Quarto renders notebook-backed examples while building the book, so local HTML and PDF renders need this slim Python environment. It covers the packages used by the executable chapter examples without pulling in the full research stack.

### 6. Install the full Python environment (optional)

```bash
pip install -r requirements.txt
```

Install the full environment only when you want to run the data-processing scripts, embedding pipelines, or other non-build Python tooling in the repository.

### 7. Install R dependencies (one-time setup)

```bash
R -e "renv::restore()"
```

Or interactively:
```bash
R
> renv::restore()
> q()
```

### 8. Install Quarto extensions (one-time setup)

The book uses two Quarto extensions for interactive Python code and pseudocode rendering:

```bash
cd aims
quarto add coatless/quarto-pyodide --no-prompt
quarto add leovan/quarto-pseudocode --no-prompt
```

This will create an `_extensions/` directory with the required extensions.

## Building the Book

### Preview (development)

To preview the book locally with live reload:

```bash
quarto preview
```

Then open http://localhost:4200/ in your browser.

For the fastest edit loop, stay on `quarto preview` or `quarto render --to html`. The PDF render is the slow path and is only needed when you want the deployable artifact or need to check the print output.

### Build HTML

```bash
quarto render --to html
```

This local HTML build omits the PDF download button on purpose, so the output is self-consistent even when you have not rendered the PDF.

### Build PDF

```bash
quarto render --to pdf
```

### Build both HTML and PDF

```bash
quarto render --to pdf
quarto render --to html --profile deploy --no-clean
```

The built website will be stored in the `_book/` folder.

### Test The Deployable Artifact Locally

To inspect the exact static files you would publish:

```bash
quarto render --to pdf
quarto render --to html --profile deploy --no-clean
cd _book
python -m http.server 8000
```

Then open http://127.0.0.1:8000/ in your browser.

## Publishing

After completing edits and building:

```bash
git add .
git commit -m "your commit message"
git push origin main
```

The GitHub Actions workflow will automatically build and publish to GitHub Pages. The Stanford proxy serves that published book at `https://aimslab.stanford.edu/textbook/`.

Regular pushes to `main` now publish the HTML site and reuse the most recently published PDF from `gh-pages` when one exists. This keeps the PDF download available without rebuilding the PDF on every content edit.

When you want to refresh the published PDF, run the `Render and Publish` workflow manually from GitHub Actions with `rebuild_pdf=true`.

## Server Deployment (Stanford)

To build and deploy on the Stanford server without compiling locally:

### One-time setup

1. Create a conda environment with Quarto:

```bash
conda create -n aims python=3.11 quarto=1.9.36 -c conda-forge
conda activate aims
pip install -r requirements-build.txt
```

Install `requirements.txt` separately only if you also need the local data-processing, embedding, narration, or other non-build Python tooling.

2. Clone the repository to the server:

```bash
cd /lfs/skampere1/0/sttruong
git clone https://github.com/aims-foundations/aims
```

### Deploying updates

Run the deploy script:

```bash
./deploy.sh
```

This will:
1. Use the current local checkout
2. Build PDF book
3. Build HTML book
4. Sync everything to `/afs/cs/group/aimslab/www/textbook/`

By default the script deploys to `/afs/cs/group/aimslab/www/textbook/`. Override this with `AIMS_DEPLOY_TARGET=/your/path ./deploy.sh`. If you need a non-default conda environment, set `AIMS_CONDA_ENV=<env-name>`. If that environment is unavailable, the script falls back to the current shell environment.

For faster HTML-only iteration on the server, skip the PDF step:

```bash
AIMS_BUILD_PDF=0 ./deploy.sh
```

That preserves the current default behavior while giving you a faster path when you do not need to refresh the PDF. If `_book/AI-Measurement-Science.pdf` is missing, the deploy-profile HTML build will omit the PDF download link.

Alternatively, run the steps manually:

```bash
eval "$(conda shell.bash hook)"
conda activate aims
cd /lfs/skampere1/0/sttruong/aims
git pull

# Build PDF first
quarto render --to pdf

# Build HTML without clearing the PDF output
quarto render --to html --profile deploy --no-clean

# Sync all to the textbook directory
rsync -av --delete _book/ /afs/cs/group/aimslab/www/textbook/
```

**Note:** Render PDF first and HTML second with `--profile deploy --no-clean`; otherwise the HTML build will not include the PDF download link, and omitting `--no-clean` will clear `_book/`.

## Chapter Videos (Animations)

Each chapter includes an optional animated video overview built with [Manim Community Edition](https://www.manim.community/) (3Blue1Brown-style). All animation source files, scripts, and tooling live in the `animations/` directory.

See [`animations/HOWTO.md`](animations/HOWTO.md) for the full guide on creating videos for new chapters.

### Quick start

```bash
# Install Manim (one-time, requires conda for system libs)
conda install -y -c conda-forge pango cairo pkg-config ffmpeg
pip install manim

# Render a single animation (1080p60)
manim -qh --disable_caching animations/icc_models.py ICCModels

# Stitch all chapter clips into a final video with background music
bash animations/stitch.sh --music animations/music/chopin_nocturne_op9_no2.mp3
```

### File structure

| Path | Description |
|------|-------------|
| `animations/HOWTO.md` | Full guide for creating chapter videos |
| `animations/<concept>.py` | Manim scenes for individual concepts |
| `animations/section_titles.py` | Title card scenes (opening, parts, closing) |
| `animations/stitch.sh` | ffmpeg stitching script (crossfade + music) |
| `animations/music/` | Background music tracks (CC0 / public domain) |
| `animations/script.md` | Narration script with animation cues |
| `animations/animation.md` | Animation plan with storyboards |

Videos are embedded in chapters using Quarto's `{{< video >}}` shortcode.

## Troubleshooting

- **Quarto version issues:** Ensure you have Quarto >= 1.9.36. Check with `quarto --version`.
- **Missing filter errors (pyodide, pseudocode):** Install the required Quarto extensions:
  ```bash
  quarto add coatless/quarto-pyodide --no-prompt
  quarto add leovan/quarto-pseudocode --no-prompt
  ```
- **R package issues:** Run `renv::restore()` in R to reinstall dependencies.
- **Python render issues:** Run `pip install -r requirements-build.txt` to reinstall the slim build environment.
- **Python tooling issues:** Run `pip install -r requirements.txt` if the non-build data-processing or embedding scripts are failing.
- **LaTeX/PDF issues:** The CI uses TinyTeX. Install with `quarto install tinytex`.

## Reproducibility

This book is designed to be reproducible. Key version recommendations:

- **Python:** 3.11 (used in CI)
- **R:** 4.3.1 (locked in renv.lock)
- **Quarto:** 1.9.36

### Clearing caches

If you upgrade dependencies or encounter stale results, clear the caches:

```bash
rm -rf _freeze/ src/.jupyter_cache/ _book/
quarto render --to html
```

## License

This book is licensed [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
