import argparse
import glob
import os

import pandas as pd
import torch
from easse.easse import bleu
from bert_score import BERTScorer
from sentence_transformers import SentenceTransformer, util


def calculate_bleu(reference: str, prediction: str) -> float:
    """Corpus-level BLEU for a single (reference, prediction) pair."""
    return bleu.corpus_bleu(refs_sents=[[str(reference)]], sys_sents=[str(prediction)])


def calculate_camembert_similarity(model, reference: str, prediction: str) -> float:
    """CamemBERT cosine similarity between two French texts."""
    ref_emb = model.encode(str(reference), convert_to_tensor=True)
    pred_emb = model.encode(str(prediction), convert_to_tensor=True)
    return float(torch.mean(util.pytorch_cos_sim(pred_emb, ref_emb)).item())


def find_complex_column(df, language: str):
    """Return the source-side column for a language ('{Lang} Complex' or '{Lang} Translated')."""
    for suffix in ("Complex", "Translated"):
        col = f"{language} {suffix}"
        if col in df.columns:
            return col
    return None


def run(input_dir="input", output_dir="automatic metrics results",
        mbert_model="bert-base-multilingual-cased",
        camembert_model="dangvantuan/sentence-camembert-base"):

    os.makedirs(output_dir, exist_ok=True)
    # Multilingual BERTScore handles the cross-lingual reference/candidate pairs.
    scorer = BERTScorer(model_type=mbert_model)
    # Lazy-load the monolingual scorers only when a corpus needs them.
    english_scorer = None
    camembert = None

    for input_path in sorted(glob.glob(os.path.join(input_dir, "*.xlsx"))):
        name = os.path.splitext(os.path.basename(input_path))[0]
        # Read the first sheet, whatever it is named.
        df = pd.read_excel(input_path, sheet_name=0)

        en_complex = find_complex_column(df, "English")
        fr_complex = find_complex_column(df, "French")
        en_simple = "English Simplified" if "English Simplified" in df.columns else None
        fr_simple = "French Simplified" if "French Simplified" in df.columns else None

        # BLEU per language: complex/translated vs simplified (same target language).
        if en_complex and en_simple:
            df["bleu_english"] = df.apply(
                lambda row: calculate_bleu(row[en_complex], row[en_simple]), axis=1
            )
        if fr_complex and fr_simple:
            df["bleu_french"] = df.apply(
                lambda row: calculate_bleu(row[fr_complex], row[fr_simple]), axis=1
            )

        # mBERT (cross-lingual): each simplification scored against the other-language complex text.
        if en_complex and fr_simple:
            df["simplification_mbert_fr"] = scorer.score(
                cands=df[fr_simple].astype(str).tolist(),
                refs=df[en_complex].astype(str).tolist(),
            )[2]
        if fr_complex and en_simple:
            df["simplification_mbert_en"] = scorer.score(
                cands=df[en_simple].astype(str).tolist(),
                refs=df[fr_complex].astype(str).tolist(),
            )[2]

        # Monolingual simplification quality: complex/translated vs simplified, same language.
        if en_complex and en_simple:
            if english_scorer is None:
                english_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
            df["bert_score_english"] = english_scorer.score(
                cands=df[en_simple].astype(str).tolist(),
                refs=df[en_complex].astype(str).tolist(),
            )[2]
        if fr_complex and fr_simple:
            if camembert is None:
                camembert = SentenceTransformer(camembert_model)
            df["camembert_score_french"] = df.apply(
                lambda row: calculate_camembert_similarity(camembert, row[fr_complex], row[fr_simple]),
                axis=1,
            )

        out_path = os.path.join(output_dir, f"{name} with automatic metrics.xlsx")
        df.to_excel(out_path, index=False)
        print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute BLEU, BERTScore, CamemBERT Score and mBERT for each corpus."
    )
    parser.add_argument("--input-dir", default="input")
    parser.add_argument("--output-dir", default="automatic metrics results")
    parser.add_argument("--mbert-model", default="bert-base-multilingual-cased",
                        help="Multilingual model for BERTScore.")
    parser.add_argument("--camembert-model", default="dangvantuan/sentence-camembert-base",
                        help="SentenceTransformer model for French similarity.")
    args = parser.parse_args()

    run(args.input_dir, args.output_dir, args.mbert_model, args.camembert_model)


if __name__ == "__main__":
    main()