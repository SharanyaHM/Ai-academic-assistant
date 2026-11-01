# AI Academic Assistant

A Streamlit application focused on processing uploaded course documents, retrieving relevant passages, and generating mark-calibrated answers with source document and page references.

## Requirements

- Python 3.11 or newer
- pip
- Semantic retrieval runs locally with Sentence Transformers and does not need an API key
- A Gemini API key is optional and used only for online answer generation and evaluation

## Local setup

Run these commands from the project directory.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
python -m streamlit run app.py
```

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
python -m streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501). To stop the server, press `Ctrl+C` in the terminal.

## Configuration and secrets

The first semantic indexing run downloads `sentence-transformers/all-MiniLM-L6-v2` into the local model cache. Later runs reuse the cached model. Choose **Semantic Search (local Sentence Transformer)** or **TF-IDF Keyword Search (local)** in Settings & Controls. These are independent retrieval methods; semantic mode does not fall back to TF-IDF when the local model fails.

For Gemini-backed answer generation and evaluation, set `GEMINI_API_KEY` in `.env`:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

For hosted Streamlit deployments, configure the key in the host's secrets manager (or `.streamlit/secrets.toml` for local testing):

```toml
GEMINI_API_KEY = "your_gemini_api_key_here"
```

Never commit `.env` or `.streamlit/secrets.toml`. The repository ignores both; `.env.example` contains only a placeholder.

## Run the application

```bash
python -m streamlit run app.py
```

Upload PDF or TXT course material, choose the retrieval method, wait for indexing to complete, enter a question, select a mark value, and choose **Get Answer**. Retrieved source passages, source filenames, page numbers, and the method-specific similarity score are available under **Verified Citations from PDF**.

Uploads and the SQLite history are stored locally under `data/`. These files are excluded from Git. Back them up before removing local data. The app currently has no user authentication or per-user storage isolation; keep it on a trusted network or use a private deployment for non-public course material.

## Optional demo material

The sample PDFs are generated locally and are not committed to the repository. To create them:

```bash
python generate_sample_pdf.py
python generate_cn_sample_pdf.py
```

The generated sample PDFs are available for local testing; they are not loaded into the app automatically.

## Tests

Several integration tests expect generated sample PDFs. Create them first using the commands above, then run the tests from the project directory:

```bash
python test_evaluator_scheme.py
python test_marks_based_generation.py
python test_pdf_isolation.py
python test_system.py
python verify_features.py
python -m unittest test_semantic_retrieval.py
```

These tests may write to the local upload directory and SQLite database. Use a disposable checkout when preserving existing local records matters.

## Deployment notes

The repository includes Streamlit theme/server defaults in `.streamlit/config.toml`, and dependencies are listed in `requirements.txt`. A typical Streamlit host can use `app.py` as its entry point and install from `requirements.txt`; configure `GEMINI_API_KEY` in the host's secrets manager when online generation is needed.

This project has not been deployed or verified on a hosting provider. The current upload and history storage is local filesystem/SQLite storage, so a host with ephemeral storage will not preserve user documents or history across restarts. Add durable storage, access control, and per-user isolation before exposing it to multiple users or the public internet.

## Project modules

- `app.py` — Streamlit UI, session state, and workflow orchestration
- `pdf_processor.py` — PDF/TXT extraction and page-aware chunking
- `rag_engine.py` — document retrieval and source metadata
- `generator.py` — grounded answers, formatting, diagrams, and rubrics
- `evaluator.py` — student answer evaluation
- `database.py` — SQLite persistence
- `config.py` — paths, model names, and academic rubrics
