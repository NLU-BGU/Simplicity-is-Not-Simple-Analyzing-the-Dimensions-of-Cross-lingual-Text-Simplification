from deep_translator import GoogleTranslator
import pandas as pd
import argparse
import os
from typing import Iterable, Optional

def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """Translate a single string using Google Translate (via deep-translator).

    Args:
        text: The text to translate.
        source_lang: Source language code (e.g., "en", "fr").
        target_lang: Target language code (e.g., "en", "fr").

    Returns:
        The translated text.
    """
    translator = GoogleTranslator(source=source_lang, target=target_lang)
    return translator.translate(text)


DEFAULT_FILES = [
    "Clear",
    "WikiLarge FR",
    "asset",
    "MultiCochrane",
    "WikiAuto",
]


def run(files: Iterable[str], input_dir: str = "input", output_dir: str = "output") -> None:
    """Translate dataset columns and write translated XLSX outputs.

    Args:
        files: Iterable of dataset basenames (without extension).
        input_dir: Directory containing `*.xlsx` inputs.
        output_dir: Directory to write translated outputs.
    """
    os.makedirs(output_dir, exist_ok=True)

    for name in files:
        df = pd.read_excel(os.path.join(input_dir, f"{name}.xlsx"))

        if name in ["Clear", "WikiLarge FR"]:
            df["English Translated"] = df.apply(
                lambda row: translate_text(str(row["French Complex"]), "fr", "en"),
                axis=1,
            )
            df["English Simplified"] = df.apply(
                lambda row: translate_text(str(row["French Simplified"]), "fr", "en"),
                axis=1,
            )
        elif name in ["asset", "WikiAuto", "MultiCochrane"]:
            df["French Translated"] = df.apply(
                lambda row: translate_text(str(row["English Complex"]), "en", "fr"),
                axis=1,
            )
            if name != "MultiCochrane":
                df["French Simplified"] = df.apply(
                    lambda row: translate_text(str(row["English Simplified"]), "en", "fr"),
                    axis=1,
                )

        df.to_excel(os.path.join(output_dir, f"{name} translated.xlsx"), index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate dataset columns via Google Translate.")
    parser.add_argument("--input-dir", default="input")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument(
        "--files",
        default=None,
        help="Comma-separated dataset names (default: built-in list).",
    )
    args = parser.parse_args()

    files = DEFAULT_FILES
    run(files=files, input_dir=args.input_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()