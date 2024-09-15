# Text Matcher CLI

This project provides a command-line interface (CLI) tool for training a text vectorizer and finding the best matching
document from a set of documents based on a query. The CLI supports multiple vectorization methods and distance metrics
for document comparison.

## Features

- **train**: train a text vectorizer using different types (`CountVectorizer`, `HashingVectorizer`, `TfidfVectorizer`).
- **pick-best**: Find the best matching document from a set of documents given a query.
- Supports distance metrics such as (`cosine`, `Euclidean`, and `Manhattan`) distances.

## Installation

### Requirements

- Python 3.12+
- Required Python libraries (see `requirements.txt`)

```bash
pip install -r requirements.txt
```

## Usage

there are two commands available:

#### 1. train

train a text vectorizer.

- **Arguments**:
    - `vectorizer_type`: type of vectorizer to use (`count`, `hashing`, `tfidf`). **[default: count]**
    - `train_file_path`: path to the data file with Wikipedia URLs . **[default: data/train.csv]**
    - `output_model_path`: path to save the trained model. **[default: vectorizer_model.pkl]**
    - `vectorizer_params: additional parameters for the vectorizer in json format`. (default params: see below)

example usage:

```bash
python -m text_matcher.cli train
```

or specify parameters:

```bash
python -m text_matcher.cli train --vectorizer-type tfidf --train-file-path data/train.csv --output-model-path vectorizer_model.pkl --vectorizer-params '{"max_df": 1, "min_df": 1, "binary": true}' 


```

#### 2. pick-best

find the best matching document from a set of documents given a query.

- **Arguments**:
    - `query_url`: Query URL to find the best match for **[required]**
    - `test_file_path`:  Path to the CSV file with test document URLs **[default: data/test.csv]**
    - `model_path`:  Path to the saved vectorizer model **[default: vectorizer_model.pkl]**
    - `distance_metric`:  Distance metric: cosine, euclidean, manhattan **[default: cosine]**

example usage:

if you want to match single document:

```bash
python -m text_matcher.cli pick-best https://pl.wikipedia.org/wiki/ED-209 --documents-path data/test.csv --vectorizer-path vectorizer_model.pkl --distance-metric cosine
```

if you want to match multiple documents:

```bash
python -m text_matcher.cli pick-best data/queries.csv --documents-path data/test.csv --vectorizer-path vectorizer_model.pkl --distance-metric cosine
```

### Default Vectorizer Parameters

```json
{
  "all": {
    "lowercase": true,
    "token_pattern": "r'\b\w\w+\b'",
    "ngram_range": [
      1,
      1
    ],
    "analyzer": "word",
    "binary": false
  },
  "count": {
    "max_df": 1,
    "min_df": 1,
    "max_features": null
  },
  "hashing": {
    "n_features": 1048576,
    "alternate_sign": true
  },
  "tfidf": {
    "norm": "l2",
    "smooth_idf": true,
    "sublinear_tf": false,
    "use_idf": true
  }
}
```

### get help

If you need help, you can use the `--help` flag to get more information about the available options for each command
i.e.,

```bash
python -m text_matcher.cli pick-best --help
````

## Running Tests

```bash
pytest tests/
```