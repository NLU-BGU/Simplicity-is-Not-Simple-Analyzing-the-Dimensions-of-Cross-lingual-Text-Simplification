# Cross-lingual Text Simplification in English and French

Code for the paper **"Translate or Simplify First: An Analysis of Cross-lingual Text Simplification in English and French"**.

Four standalone scripts, each with a CLI (`python <script>.py --help`).

## Data
The input data (complex texts and their simplified versions) is available here:
[Google Drive](https://drive.google.com/drive/folders/1rzH292aF3MRn7WMEWy7i12XfQ7PcLvfo?usp=drive_link)

For the literature texts (*Around the World in 80 Days* and *Dr. Jekyll and Mr. Hyde*) we publish only the complex texts (complex and translated); the simplified versions are withheld due to copyright.

## Setup
```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
```
`linguistic_features.py --download-resources` fetches the required NLTK corpora on first run.

## `texts_translations.py`
Translates dataset columns between English and French via Google Translate (`deep-translator`). Writes `<name> translated.xlsx`.

## `automatic_metrics.py`
Computes automatic metrics per corpus: BLEU (per language), monolingual BERTScore (English) / CamemBERT similarity (French), and cross-lingual mBERT. Writes `<corpus> with automatic metrics.xlsx`.

## `linguistic_features.py`
Computes ~30 lexical/syntactic complexity features (spaCy / NLTK / textstat) and outputs the per-row complex−simplified **difference** for each CLTS/MLTS pair. Writes `<corpus> linguistic features.xlsx`.

## `human_annotation_correlation.py`
Spearman correlations within each annotation sheet: (1) human simplicity vs each linguistic-feature difference, (2) human meaning-preservation vs each automatic metric. Writes `<corpus> correlations.xlsx`.
