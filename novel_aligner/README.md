# Novel Aligner - Japanese/English Light Novel Chapter Alignment System

A production-ready system for automatically aligning Japanese and English light novel chapters using semantic matching and dynamic programming sequence alignment.

## Project Structure

```
novel_aligner/
├── core/
│   ├── __init__.py
│   ├── file_loader.py
│   ├── chapter_fingerprint.py
│   ├── embedding_engine.py
│   ├── similarity_matrix.py
│   └── sequence_aligner.py
├── api/
│   ├── __init__.py
│   └── fastapi_server.py
├── database/
│   ├── __init__.py
│   └── db_manager.py
├── tests/
│   ├── __init__.py
│   └── test_alignment.py
├── data/
│   └── (sample data)
├── cli.py
├── graph_builder.py
├── config.py
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download ja_core_news_sm
```

## Usage

```bash
# Align a novel
python cli.py align /path/to/NovelName

# Rebuild embeddings
python cli.py rebuild-embeddings /path/to/NovelName

# Export graph
python cli.py export-graph /path/to/NovelName output.json

# Validate
python cli.py validate /path/to/NovelName

# Stats
python cli.py stats /path/to/NovelName
```

## Configuration

Edit `config.py` to customize:
- Embedding model
- Similarity weights
- Chapter sampling size
- Confidence thresholds

## API Endpoints

- `GET /novels` - List all novels
- `GET /novel/{id}` - Get novel details
- `GET /novel/{id}/relations` - Get chapter relations
- `GET /novel/{id}/unmatched` - Get unmatched chapters
- `POST /novel/{id}/align` - Trigger alignment
- `GET /chapter/{id}` - Get chapter details
- `GET /health` - Health check
