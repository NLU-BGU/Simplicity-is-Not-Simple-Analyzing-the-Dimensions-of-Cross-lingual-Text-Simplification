import glob
import os

import pandas as pd
from scipy.stats import spearmanr

N_ROWS = 70

# Annotation columns.
SIMPLER_COL = "Is Simplification simpler than Original ? (from -2 which is more complex to 2 which is simpler)"
ADD_SCALE_COL = "Does Simplification add information compared to Original ? Scale"
REMOVE_SCALE_COL = "Does Simplification remove information compared to Original ? Scale"

# Automatic-metric columns that may appear on an annotation sheet. Each sheet
# carries the metric(s) computed on its own directional text pair (e.g. EO_FT's
# mbert_score, FT_FS's bleu_score/camembert_score); correlation #2 pairs those
# with the same sheet's meaning-preservation annotation.
METRIC_COLS = ["mbert_score", "bleu_score", "bert_score", "camembert_score"]

# The annotation filename suffix shared by every corpus workbook.
ANNOTATION_SUFFIX = " annotations.xlsx"

# Which annotation sheets carry each corpus's real simplifications (correlation #1
# runs only on these CLTS/MLTS pairs). Maps the corpus name discovered from the
# annotation filename -> {sheet code: feature workbook stem}, where the stem
# resolves to "{stem} linguistic features.xlsx".
ENGLISH_PAIRS = {"EO_FS", "FT_FS"}          # English-original corpora
FRENCH_PAIRS = {"FO_ES", "ET_ES"}           # French-original corpora
FEATURE_SHEETS = {
    "asset": ENGLISH_PAIRS,
    "wikiauto": ENGLISH_PAIRS,
    "multicochrane": ENGLISH_PAIRS,
    "the strange case of dr. jekyll and mr. hyde": ENGLISH_PAIRS,
    "clear": FRENCH_PAIRS,
    "wikilargefr": FRENCH_PAIRS,
    "around the world in 80 days": FRENCH_PAIRS,
    "EU culture": {
        "FO_ES": "EU culture ES",
        "EO_ES": "EU culture ES",
        "EO_FS": "EU culture FS",
        "FO_FS": "EU culture FS",
    },
    "EU human rights": {
        "FO_ES": "EU human rights ES",
        "EO_ES": "EU human rights ES",
        "EO_FS": "EU human rights FS",
        "FO_FS": "EU human rights FS",
    },
}

# EU corpora keep both simplified languages in one annotation file but are written
# out as two correlation files, one per simplified language. Maps corpus name ->
# {output name suffix: predicate selecting that language's sheets}. A sheet's
# simplified language is the target half of its code (e.g. "EO_ES" -> "ES").
EU_OUTPUT_SPLIT = {
    "EU culture", "EU human rights",
}


def _feature_label(sheet_code: str) -> str:
    """Derive the feature sheet (CLTS/MLTS) from an annotation sheet code.

    Codes look like "EO_FS": {source lang}{O/T} _ {target lang}{S}. CLTS is the
    cross-lingual pair (source language != target language); MLTS is the
    same-language pair (source language == target language). Returns None for
    codes that aren't a feature pair (e.g. "EO_FO", which has no simplification).
    """
    if "_" not in sheet_code:
        return None
    src, tgt = sheet_code.split("_", 1)
    if not tgt.endswith("S"):  # not a simplified pair
        return None
    return "CLTS" if src[0] != tgt[0] else "MLTS"


def _spearman_rows(annotation: pd.Series, table: pd.DataFrame, corr_type: str) -> list:
    """Spearman of `annotation` vs each numeric column of `table`.

    NaNs are dropped pairwise. Returns a list of result dicts.
    """
    rows = []
    for col in table.columns:
        values = pd.to_numeric(table[col], errors="coerce")
        paired = pd.concat([annotation, values], axis=1).dropna()
        if len(paired) < 3 or paired.iloc[:, 0].nunique() < 2 or paired.iloc[:, 1].nunique() < 2:
            r, p = float("nan"), float("nan")
        else:
            r, p = spearmanr(paired.iloc[:, 0], paired.iloc[:, 1])
        rows.append({"correlation": corr_type, "variable": col,
                     "spearman_r": r, "p_value": p, "n": len(paired)})
    return rows


def _meaning_preservation(df: pd.DataFrame) -> pd.Series:
    """Recompute Meaning Preservation Scale = 5 - (add_scale + remove_scale) / 2."""
    add = pd.to_numeric(df[ADD_SCALE_COL], errors="coerce")
    remove = pd.to_numeric(df[REMOVE_SCALE_COL], errors="coerce")
    return 5 - (add + remove) / 2


def _simpler_annotation(df: pd.DataFrame) -> pd.Series:
    """The simplicity annotation: the named column, else the 3rd column."""
    if SIMPLER_COL in df.columns:
        return pd.to_numeric(df[SIMPLER_COL], errors="coerce")
    return pd.to_numeric(df.iloc[:, 2], errors="coerce")


def _feature_stems(corpus: str) -> dict:
    """Map each CLTS/MLTS annotation sheet code to its feature workbook stem.

    EU corpora declare a {code: stem} dict (features split by simplified language);
    every other corpus declares a set of codes that all live in a single workbook
    named after the corpus. Returns {} for corpora with no declared feature sheets.
    """
    spec = FEATURE_SHEETS.get(corpus)
    if spec is None:
        return {}
    if isinstance(spec, dict):
        return dict(spec)
    return {code: corpus for code in spec}


def _load_feature_sheets(feature_dir: str, corpus: str) -> dict:
    """Load the feature table for each declared CLTS/MLTS sheet of a corpus.

    Returns {sheet_code: features_df}. Skips feature workbooks that don't exist.
    """
    by_code = {}
    for code, stem in _feature_stems(corpus).items():
        label = _feature_label(code)
        path = os.path.join(feature_dir, f"{stem} linguistic features.xlsx")
        if not os.path.exists(path):
            print(f"  [warn] missing feature file, skipping correlation #1 for {code}: {path}")
            continue
        try:
            feats = pd.read_excel(path, sheet_name=label).head(N_ROWS)
        except (ValueError, OSError) as e:
            # Unreadable/empty/partly-written workbook (e.g. still being generated).
            print(f"  [warn] unreadable feature file, skipping correlation #1 for {code}: {path} ({e})")
            continue
        # Keep only numeric feature columns (drops the leading text columns).
        by_code[code] = feats.select_dtypes(include="number")
    return by_code


def _correlate_sheet(df: pd.DataFrame, feature_table) -> pd.DataFrame:
    """Both correlations for one annotation sheet, as a result table.

    Both correlations stay within this single sheet (its own directional text
    pair): correlation #1 vs its CLTS/MLTS feature differences, correlation #2 vs
    the metric column(s) computed on the same pair.

    `feature_table` is the sheet's CLTS/MLTS feature DataFrame for correlation #1,
    or None to skip it (a sheet that is not a declared CLTS/MLTS pair).
    """
    results = []

    # #1: this sheet's simplicity annotation vs its linguistic feature differences.
    if feature_table is not None:
        simpler = _simpler_annotation(df).reset_index(drop=True)
        feats = feature_table.reset_index(drop=True)
        # Align by row order on the shared length.
        n = min(len(simpler), len(feats))
        results += _spearman_rows(simpler.iloc[:n].reset_index(drop=True),
                                  feats.iloc[:n].reset_index(drop=True),
                                  "is_simpler_vs_feature")

    # #2: this sheet's meaning preservation vs the metric(s) on the same sheet.
    metric_cols = [c for c in METRIC_COLS if c in df.columns]
    if metric_cols:
        meaning = _meaning_preservation(df).reset_index(drop=True)
        results += _spearman_rows(meaning, df[metric_cols].reset_index(drop=True),
                                  "meaning_vs_metric")

    return pd.DataFrame(
        results, columns=["correlation", "variable", "spearman_r", "p_value", "n"]
    )


def _output_groups(corpus: str, sheet_names) -> dict:
    """Group a corpus's annotation sheets into output files.

    Most corpora produce one file ({corpus}). EU corpora are split by simplified
    language into two files, keyed by the sheet's target half ("ES" -> English,
    "FS" -> French); Returns {output name: [sheet codes]}.
    """
    if corpus not in EU_OUTPUT_SPLIT:
        return {corpus: list(sheet_names)}

    en, fr = [], []
    for sheet in sheet_names:
        if sheet.endswith("ES"):
            en.append(sheet)
        elif sheet.endswith("FS"):
            fr.append(sheet)
        else:  # translation-only sheets (e.g. "EO_FO") go in both files
            en.append(sheet)
            fr.append(sheet)
    return {f"{corpus} ES": en, f"{corpus} FS": fr}


def run(annotation_dir="human annotation",
        feature_dir="linguistic features output",
        output_dir="human annotation correlation"):
    """Build per-corpus correlation workbooks for every annotation file found.

    Correlation #1 reads linguistic features from `feature_dir`; correlation #2
    reads each annotation sheet's own in-sheet metric column(s).
    """
    os.makedirs(output_dir, exist_ok=True)

    pattern = os.path.join(annotation_dir, f"*{ANNOTATION_SUFFIX}")
    for annotation_path in sorted(glob.glob(pattern)):
        name = os.path.basename(annotation_path)[: -len(ANNOTATION_SUFFIX)]
        xls = pd.ExcelFile(annotation_path)
        feature_by_code = _load_feature_sheets(feature_dir, name)

        # Correlate every sheet once, then route the results into output files.
        tables = {}
        for sheet in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet).head(N_ROWS)
            tables[sheet] = _correlate_sheet(df, feature_by_code.get(sheet))

        for output_name, sheets in _output_groups(name, xls.sheet_names).items():
            out_path = os.path.join(output_dir, f"{output_name} correlations.xlsx")
            with pd.ExcelWriter(out_path) as writer:
                for sheet in sheets:
                    tables[sheet].to_excel(writer, sheet_name=sheet, index=False)
            print(f"Wrote {out_path}")


if __name__ == "__main__":
    run()
