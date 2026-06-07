import argparse
import os
from functools import lru_cache

import nltk
import numpy as np
import pandas as pd
import pyphen
import spacy
import textstat
from nltk.corpus import wordnet, words as nltk_words
from nltk.tokenize import sent_tokenize, word_tokenize


def ensure_resources(download: bool = False) -> None:
    """Ensure required NLP resources are available.

    By default this function does **not** download anything (GitHub-friendly).
    If you pass `download=True`, it will attempt to download missing resources.

    Args:
        download: If True, download missing spaCy / NLTK resources.
    """
    if not download:
        return

    # NLTK resources (safe to call; no-op if present)
    for pkg in [
        "punkt",
        "averaged_perceptron_tagger",
        "wordnet",
        "words",
        "omw-1.4",
    ]:
        nltk.download(pkg)

    # spaCy French model (download if missing)
    try:
        spacy.load("fr_core_news_sm")
    except OSError:
        spacy.cli.download("fr_core_news_sm")

    # spaCy English model (download if missing)
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        spacy.cli.download("en_core_web_sm")


@lru_cache(maxsize=1)
def get_french_wordnet_words() -> set[str]:
    """Get a set of French words from Open Multilingual Wordnet.

    Returns:
        A lowercase set of French lemma strings.
    """
    french_words: set[str] = set()
    for synset in wordnet.all_synsets():
        for lemma in synset.lemmas(lang="fra"):
            french_words.add(lemma.name().lower())
    return french_words
    
class TextComplexityAnalyzer:
    """Compute a set of lexical/syntactic complexity metrics for a text."""

    def __init__(self, text: str, language: str):
        """
        Args:
            text: Input text.
            language: "english" or "french" (lowercase).
        """
        self.language = language
        self.text = text

        if language == "english":
            self.nlp = spacy.load("en_core_web_sm")
        elif language == "french":
            self.nlp = spacy.load("fr_core_news_sm")
        else:
            raise ValueError("language must be 'english' or 'french'")

        self.doc = self.nlp(text)
        self.tokens = word_tokenize(text)
        self.sentences = sent_tokenize(text)

        # Precompute some analyses
        self.pos_tags = nltk.pos_tag(self.tokens)
        self.clean_tokens = [
            str(token) for token in self.doc if (not token.is_punct) and (not token.is_space)
        ]


    def lexical_richness(self):
        """Average lexical richness (unique words / total words)"""
        lowered_words = [word.lower() for word in self.clean_tokens]
        unique_words = len(set(lowered_words))
        return unique_words/len(self.clean_tokens) if len(self.clean_tokens)>0 else 0

    def words_before_main_verb(self):
        """Average number of words before the main verb in each sentence"""
        distances = []
        
        # Iterate through each sentence in the document
        for sent in self.doc.sents:
            # The 'root' attribute of a sentence is the token that is the root of the dependency parse tree
            root_token = sent.root
            
            # Check if the root token is a verb. It usually is, but not always.
            if root_token.pos_ == 'VERB':
                # Calculate the index of the root verb relative to the start of the sentence
                # sent[0].i is the absolute index of the first token in the sentence
                root_verb_index = root_token.i - sent[0].i
                distances.append(root_verb_index)
            else:
                main_verb_index = None
                for token in sent:
                    if token.pos_ == 'VERB':
                        main_verb_index = token.i - sent[0].i
                        break
                if main_verb_index is not None:
                    distances.append(main_verb_index)
                
        # Return the average distance
        return sum(distances) / len(distances) if distances else 0

    def entity_distance(self):
        """Calculate max distance between same entity appearances"""
        entity_positions = {}
        for ent in self.doc.ents:
            if ent.text.lower() not in entity_positions:
                entity_positions[ent.text.lower()] = []
            entity_positions[ent.text.lower()].append(ent.start)
    
        all_max_distances = []
        for positions in entity_positions.values():
            if len(positions) > 1:
                # Calculate the distance between the last and first appearance
                distance = positions[-1] - positions[0]
                all_max_distances.append(distance)
                
        # Return the maximum distance found across all entities
        return max(all_max_distances) if all_max_distances else 0

    def content_words_ratio(self):
        """Ratio of content words (nouns, verbs, adjectives, adverbs)"""
        content_pos = {'NOUN', 'VERB', 'ADJ', 'ADV'}
        content_words = sum(1 for token in self.doc if token.pos_ in content_pos and not token.is_punct)
        return content_words/len(self.clean_tokens)  if len(self.clean_tokens)>0 else 0

    def infrequent_words_ratio(self):
        """Ratio of words not in common English/French word list"""
        if self.language == "english":
            common_words = set(nltk_words.words())
            infrequent_words = sum(1 for token in self.clean_tokens if token.lower() not in common_words)
            return infrequent_words / len(self.clean_tokens) if len(self.clean_tokens) > 0 else 0
        if self.language == "french":
            common_words = get_french_wordnet_words()
            infrequent_words = sum(1 for token in self.clean_tokens if token.lower() not in common_words)
            return infrequent_words / len(self.clean_tokens) if len(self.clean_tokens) > 0 else 0
        return 0


    def long_words_ratio(self, threshold=9):
        """Ratio of words longer than threshold"""
        long_words = sum(1 for token in self.clean_tokens if len(token) > threshold)
        return long_words/len(self.clean_tokens) if len(self.clean_tokens)>0 else 0

    def modifiers_ratio(self):
        """Ratio of modifiers (adjectives, adverbs)"""
        modifier_pos = {'ADJ', 'ADV'}
        modifiers = sum(1 for token in self.doc if token.pos_ in modifier_pos)
        return modifiers/len(self.clean_tokens) if len(self.clean_tokens)>0 else 0

    def negations_ratio(self):
        """Ratio of negation words"""
        if self.language=='english':
          negation_words = {'not', 'no', 'neither', 'nor', 'none', "never", "nothing", "nowhere", "no one",
            "can't", "don't", "won't", "isn't", "wouldn't",
            "shouldn't", "couldn't", "hadn't", "doesn't", "didn't", "haven't",
            "hasn't", "weren't", "aren't", "wasn't", "mustn't"}
        elif self.language=='french':
          negation_words={'ne',"n'", "ne", "pas", "plus", "jamais", "rien", "personne", "guère",
                        "aucun", "aucune",'non', "ni"}
        negations = sum(1 for token in self.clean_tokens if token.lower() in negation_words)
        return negations/len(self.clean_tokens) if len(self.clean_tokens)>0 else 0


    def noun_phrases_ratio(self):
        """Ratio of noun phrases that contain consecutive nouns"""
        noun_phrase_count = len(list(self.doc.noun_chunks))


        return noun_phrase_count /len(self.clean_tokens) if len(self.clean_tokens)>0 else 0

    def count_past_perfect_verbs(self):
        """
        Count the ratio of past perfect verbs in a sentence.
        """
        if self.language=='english':
          past_perfect_count = 0

          for sentence in self.sentences:
              sent_tokens = word_tokenize(sentence)
              sent_tags = nltk.pos_tag(sent_tokens)

              # Count past perfect verbs
              for i in range(len(sent_tags) - 1):
                  # Check if current word is 'had' and next word is a past participle
                  if (sent_tokens[i].lower() == 'had' and sent_tags[i+1][1] == 'VBN'):
                      past_perfect_count += 1

          # Return ratio of past perfect verbs to total tokens
          past_tense = sum(1 for _, tag in self.pos_tags if tag in ['VBD', 'VBN'])
          return past_perfect_count/past_tense if past_tense!=0 else 0
        return 0

    def verb_tense_analysis(self):
        """Analyze past tense verb ratios"""
        if self.language == 'english':
            past_tense = sum(1 for _, tag in self.pos_tags if tag in ['VBD', 'VBN'])
        else:
            past_tense = sum(1 for token in self.doc if 'Tense=Past' in token.morph)
        verbs=sum(1 for token in self.doc if token.pos_ == 'VERB')
        return past_tense/verbs if verbs!=0 else 0

    def punctuation_ratio(self):
        """Ratio of punctuation marks"""
        punctuation_count = 0
        
        for token in self.doc:
            if token.is_punct:
                punctuation_count += 1
        return punctuation_count/len(self.tokens) if len(self.tokens)>0 else 0

    def relative_clauses_ratio(self):
        """Ratio of relative clauses"""

        if self.language=='english':
          relative_pronouns = {'who', 'whom', 'whose', 'which', 'that', "when", "where", "why"}
        elif self.language=='french':
          relative_pronouns={'qui', 'que', 'quoi', 'dont', 'où', 'lequel', 'laquelle', 
                             'lesquels', 'lesquelles', 'auquel', 'à laquelle', 
                             'auxquels', 'auxquelles', 'duquel', 'de laquelle', 
                             'desquels', 'desquelles'}
        relative_clauses = sum(1 for token in self.clean_tokens if token.lower() in relative_pronouns)
        return relative_clauses/len(self.clean_tokens) if len(self.clean_tokens)>0 else 0

    def third_person_pronouns_ratio(self):
        """Ratio of third-person singular pronouns"""
        if self.language=='english':
            third_person_pronouns = {'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its',
            'they', 'them', 'their', 'theirs', 'himself', 'herself',
            'itself', 'themselves'}
        elif self.language=='french':
            third_person_pronouns = {'il', 'elle', 'on', 'ils', 'elles', 'ce', 'c',
            'le', 'la', 'les', 'l','lui', 'leur','se', 's','eux', 'soi','y', 'en',
            'celui', 'celle', 'ceux', 'celles'}
        pronouns = sum(1 for token in self.clean_tokens if token.lower() in third_person_pronouns)
        return pronouns / len(self.clean_tokens) if len(self.clean_tokens)>0 else 0


    def unique_entities_ratio(self):
        """Ratio of unique entities"""
        unique_entities = len(set(ent.text.lower() for ent in self.doc.ents))
        return unique_entities

    def readability_metrics(self):
        """Various readability metrics"""
        return {
            'flesch_reading_ease': textstat.flesch_reading_ease(self.text),
            'flesch_kincaid_grade': textstat.flesch_kincaid_grade(self.text)
        }

    def sentences_count_ratio(self):
        """Ratio of sentences (few sentences)"""
        return len(self.sentences)

    def words_containing_more_then_8_chars(self, threshold=8):
        """Ratio of words containing more than eight characters"""
        long_words = sum(1 for token in self.clean_tokens if len(token) > threshold)
        return long_words/len(self.clean_tokens) if len(self.clean_tokens)>0 else 0

  

    def words_per_sentence(self):
        """Average number of words per sentence"""
        words_per_sentence = [len(word_tokenize(sentence)) for sentence in self.sentences]
        return np.mean(words_per_sentence)/len(self.clean_tokens) if len(self.clean_tokens)>0 else 0


    def consecutive_entity_distance(self):
        """Average distance between consecutive entities"""
        entity_indices = [ent.start for ent in self.doc.ents]
            
        if len(entity_indices) < 2:
            return 0
        
        # Calculate the distances between consecutive entity indices
        distances = np.diff(entity_indices)
        
        return np.mean(distances)


    def entity_metrics(self):
        """Calculate various entity-related metrics"""
        entities = list(self.doc.ents)

        # Unique entities metrics
        unique_entities = len(set(ent.text.lower() for ent in entities))

        # Entity to token ratio
        entity_token_ratio = len(entities) / len(self.clean_tokens) if len(self.clean_tokens)>0 else 0

        entity_positions = {}
        for ent in entities:
            # Use the entity's lowercase text as the key for the dictionary
            key = ent.text.lower()
            if key not in entity_positions:
                entity_positions[key] = []
            # Store the starting token index of the entity
            entity_positions[key].append(ent.start)
    
        avg_same_entity_distances = []
        for positions in entity_positions.values():
            if len(positions) > 1:
                # Calculate consecutive distances and their average
                distances = np.diff(positions)
                avg_same_entity_distances.append(np.mean(distances))
    
        return {
            'unique_entities_count': unique_entities/len(self.sentences) if len(self.sentences)>0 else 0,
            'entity_to_token_ratio': entity_token_ratio,
            'avg_same_entity_distance': np.mean(avg_same_entity_distances) if avg_same_entity_distances else 0,
            'unique_entities_to_total_num_of_entities': unique_entities/len(entities) if len(entities)>0 else 0
        }

    def clause_and_voice_analysis(self):
        """
        Analyze clauses, conjunctions, and voice
        """
        if self.language=='english':
            # Check for conditional clauses (very simplified)
            conditional_clauses = sum(1 for token in self.clean_tokens if token.lower() in {'if', 'unless', 'whether', 'in case'})
        elif self.language=='french':
            conditional_clauses = sum(1 for token in self.clean_tokens if token.lower() in {'si', 'à moins que', 'pourvu que', 'au cas où'})

        # Check for conjunctions
        conjunctions = sum(1 for token in self.doc if token.pos_ == 'CCONJ' or token.pos_ == 'SCONJ')
        if self.language =='english':
            # Check for passive voice (simplified)
            passive_voice = sum(1 for sent in self.doc.sents
                                if any(token.dep_ == 'auxpass' for token in sent))
        else:
            passive_voice = sum(1 for sent in self.doc.sents
                                if any(token.dep_ == 'aux:pass' and token.lemma_ == 'être' for token in sent))
        
        # Check for appositions
        appositions = sum(1 for token in self.doc if token.dep_ == 'appos')
        total_verbs = sum(1 for token in self.doc if token.pos_ == 'VERB')

        return {
            'conditional_clauses_ratio': conditional_clauses/len(self.clean_tokens) if len(self.clean_tokens)>0 else 0,
            'conjunctions_ratio': conjunctions/len(self.clean_tokens) if len(self.clean_tokens)>0 else 0,
            'passive_voice_ratio': passive_voice/total_verbs if total_verbs!=0 else 0,
            'appositions_ratio': appositions/len(self.clean_tokens) if len(self.clean_tokens)>0 else 0
        }

    def short_sentences_ratio(self, max_words=10):
        """
        Calculate the ratio of short sentences
        """
        # Count sentences with fewer than max_words
        short_sentences = sum(1 for sent in self.sentences
                               if len(word_tokenize(sent)) <= max_words)
        return short_sentences/len(self.sentences) if len(self.sentences)>0 else 0

    def calculate_avg_word_length(self) -> float:
        """
        Calculates the average length of words in the text using clean tokens.
        """
        # Calculate the average word length
        total_length = sum(len(word) for word in self.clean_tokens)
        return total_length / len(self.clean_tokens) if len(self.clean_tokens) > 0 else 0.0

    def syllable_to_word_ratio(self) -> float:
        """
        Calculates the average number of syllables per word.
        """
        if self.language == 'french':
            lang = 'fr_FR'
        elif self.language == 'english':
            lang = 'en_US'
        else:
            return 0.0
        
        # Initialize the hyphenation dictionary for the specified language
        dic = pyphen.Pyphen(lang=lang)

        # Count syllables
        total_syllables = 0
        for token in self.clean_tokens:
            # Hyphenate the word and count the syllables
            # Pyphen's .inserted() returns 'word' as 'sy-lla-ble'
            hyphenated_word = dic.inserted(token.lower())
            total_syllables += hyphenated_word.count('-') + 1

        # Calculate the ratio of syllables to words
        if len(self.clean_tokens) > 0:
            ratio = total_syllables / len(self.clean_tokens)
        else:
            ratio = 0.0  # Avoid division by zero if the text is empty

        return float(ratio)

    def get_syntactic_depth(self):
        """
        Calculates the maximum syntactic tree depth for a text.
        """
        max_depth = 0
        for token in self.doc:
            depth = 0
            current_token = token
            while current_token.head != current_token:
                current_token = current_token.head
                depth += 1
            
            # Update the maximum depth
            if depth > max_depth:
                max_depth = depth
                
        return max_depth

        

    def perform_analysis(self):
        """Aggregate all complexity metrics"""
        d= {
            'lexical_richness': self.lexical_richness(),
            'words_before_main_verb': self.words_before_main_verb(),
            'max_same_entity_distances': self.entity_distance(),
            'content_words_ratio': self.content_words_ratio(),
            'infrequent_words_ratio': self.infrequent_words_ratio(),
            'long_words_ratio': self.long_words_ratio(),
            'modifiers_ratio': self.modifiers_ratio(),
            'negations_ratio': self.negations_ratio(),
            'noun_phrases_ratio': self.noun_phrases_ratio(),
            'past_perfect_verbs': self.count_past_perfect_verbs(),
            'past_tense_verbs': self.verb_tense_analysis(),
            'punctuation_ratio': self.punctuation_ratio(),
            'relative_clauses_ratio': self.relative_clauses_ratio(),
            'sentences_number': self.sentences_count_ratio(),
            'third_person_pronouns_ratio': self.third_person_pronouns_ratio(),
            'unique_entities': self.unique_entities_ratio(),
            'words_containing_more_then_8_characters': self.words_containing_more_then_8_chars(),
            'words_per_sentence': self.words_per_sentence(),
            'consecutive_entity_distance': self.consecutive_entity_distance(),
            'flesch_reading_ease': self.readability_metrics()['flesch_reading_ease'],
            'unique_entities_average': self.entity_metrics()['unique_entities_count'],
            'avg_same_entity_distance': self.entity_metrics()['avg_same_entity_distance'],
            'entity_to_token_ratio': self.entity_metrics()['entity_to_token_ratio'],
            'flesch_kincaid_grade': self.readability_metrics()['flesch_kincaid_grade'],
            'unique_entities_to_total_num_of_entities': self.entity_metrics()['unique_entities_to_total_num_of_entities'],
            'appositions_ratio': self.clause_and_voice_analysis()['appositions_ratio'],
            'conditional_clauses_ratio': self.clause_and_voice_analysis()['conditional_clauses_ratio'],
            'conjunctions_ratio': self.clause_and_voice_analysis()['conjunctions_ratio'],
            'passive_voice_ratio': self.clause_and_voice_analysis()['passive_voice_ratio'],
            'short_sentences_ratio': self.short_sentences_ratio(),
            'syntactic_tree_depth': self.get_syntactic_depth(),
            'syllables_ratio': self.syllable_to_word_ratio(),
            'avg_word_length': self.calculate_avg_word_length()

            #'concreteness': self.concreteness_analysis()
        }
        if self.language=='french':
          del d['past_perfect_verbs']
        return d

def add_text_complexity_metrics(df, text_column, language):
    """
    Calculates linguistic complexity metrics for a given text column.
    Handles columns with single strings and lists of strings.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        text_column (str): The name of the column containing the text data.
        language (str): The language of the text ('english' or 'french').

    Returns:
        pd.DataFrame: A new DataFrame with the calculated metrics as new columns.
    """
    df_copy = df.copy()

    def process_text_or_list(text_data):
        if isinstance(text_data, list):
            if not text_data:
                # Return a series of zeros if the list is empty
                return pd.Series([0] * len(TextComplexityAnalyzer("sample", language).perform_analysis().keys()), 
                                 index=TextComplexityAnalyzer("sample", language).perform_analysis().keys())
            
            # Calculate metrics for each string in the list
            list_of_metrics_series = [
                TextComplexityAnalyzer(str(item), language).perform_analysis()
                for item in text_data
            ]

            # Convert to DataFrame and calculate the mean for each metric
            metrics_df = pd.DataFrame(list_of_metrics_series)
            return metrics_df.mean()

        else:
            # Process a single string as before
            return pd.Series(TextComplexityAnalyzer(str(text_data), language).perform_analysis())

    # Apply the new processing function to the text column
    complexity_metrics = df_copy[text_column].apply(process_text_or_list)

    # Add metrics as new columns
    df_copy = pd.concat([df_copy, complexity_metrics], axis=1)

    return df_copy



FR_SIMPLIFIED_GROUP = {  # asset, wikiauto, multicochrane, dr jekyll: "Sheet1", target = French
    "CLTS": (("Sheet1", "English Complex", "english"), ("Sheet1", "French Simplified", "french")),
    "MLTS": (("Sheet1", "French Translated", "french"), ("Sheet1", "French Simplified", "french")),
}
EN_SIMPLIFIED_GROUP = {  # wikilargefr, clear, around the world: "Sheet1", target = English
    "CLTS": (("Sheet1", "French Complex", "french"), ("Sheet1", "English Simplified", "english")),
    "MLTS": (("Sheet1", "English Translated", "english"), ("Sheet1", "English Simplified", "english")),
}
_EU_EN_SIMPLIFIED = {  # EU "English" sheet, English-simplified file
    "CLTS": (("English", "French Complex", "french"), ("English", "English Simplified", "english")),
    "MLTS": (("English", "English Complex", "english"), ("English", "English Simplified", "english")),
}
_EU_FR_SIMPLIFIED = {  # EU "English" sheet, French-simplified file
    "CLTS": (("English", "English Complex", "english"), ("English", "French Simplified", "french")),
    "MLTS": (("English", "French Complex", "french"), ("English", "French Simplified", "french")),
}

# Maps each output file basename to (input workbook filename, sheet plan).
OUTPUT_PLAN = {
    "asset": ("asset.xlsx", FR_SIMPLIFIED_GROUP),
    "wikiauto": ("wikiauto.xlsx", FR_SIMPLIFIED_GROUP),
    "multicochrane": ("multicochrane.xlsx", FR_SIMPLIFIED_GROUP),
    "the strange case of dr. jekyll and mr. hyde":
        ("The Strange Case of Dr Jeckyll and Mister Hyde.xlsx", FR_SIMPLIFIED_GROUP),
    "wikilargefr": ("wikilargefr.xlsx", EN_SIMPLIFIED_GROUP),
    "clear": ("clear.xlsx", EN_SIMPLIFIED_GROUP),
    "around the world in 80 days":
        ("Around the world in 80 days.xlsx", EN_SIMPLIFIED_GROUP),
    # EU corpora, split by simplified-language target (2 files each: ES / FS)
    "EU culture ES": ("Culture Alignements European Union.xlsx", _EU_EN_SIMPLIFIED),
    "EU culture FS": ("Culture Alignements European Union.xlsx", _EU_FR_SIMPLIFIED),
    "EU human rights ES": ("HUMAN RIGHTS Alignements European Union.xlsx", _EU_EN_SIMPLIFIED),
    "EU human rights FS": ("HUMAN RIGHTS Alignements European Union.xlsx", _EU_FR_SIMPLIFIED),
}


def compute_difference_sheet(xls, complex_side, simplified_side):
    """Per-row (complex - simplified) feature differences.

    Each side is (sheet, column, language). The two text columns are kept as the
    first two columns; only features computed for both languages are kept (e.g.
    past_perfect_verbs is English-only and is dropped for French).
    """
    (c_sheet, c_col, c_lang) = complex_side
    (s_sheet, s_col, s_lang) = simplified_side

    complex_text = pd.read_excel(xls, c_sheet)[c_col].reset_index(drop=True)
    simplified_text = pd.read_excel(xls, s_sheet)[s_col].reset_index(drop=True)

    complex_metrics = add_text_complexity_metrics(complex_text.to_frame(c_col), c_col, c_lang)
    simplified_metrics = add_text_complexity_metrics(simplified_text.to_frame(s_col), s_col, s_lang)

    features = [c for c in TextComplexityAnalyzer("sample", c_lang).perform_analysis()
                if c in complex_metrics.columns and c in simplified_metrics.columns]
    differences = (complex_metrics[features].reset_index(drop=True)
                   - simplified_metrics[features].reset_index(drop=True))

    # Disambiguate headers if both sides share a column name (monolingual MLTS).
    c_header, s_header = c_col, s_col
    if c_header == s_header:
        c_header, s_header = f"{c_col} (complex)", f"{s_col} (simplified)"

    return pd.concat(
        [complex_text.rename(c_header), simplified_text.rename(s_header), differences], axis=1
    )


def run(output_files, input_dir="input", output_dir="linguistic features output",
        download_resources=False):
    """Build the CLTS/MLTS difference workbooks for the requested output files."""
    ensure_resources(download=download_resources)
    os.makedirs(output_dir, exist_ok=True)

    for output_name in output_files:
        input_filename, sheet_plan = OUTPUT_PLAN[output_name]
        xls = pd.ExcelFile(os.path.join(input_dir, input_filename))

        out_path = os.path.join(output_dir, f"{output_name} linguistic features.xlsx")
        with pd.ExcelWriter(out_path) as writer:
            for sheet_label, (complex_side, simplified_side) in sheet_plan.items():
                sheet = compute_difference_sheet(xls, complex_side, simplified_side)
                sheet.to_excel(writer, sheet_name=sheet_label, index=False)
        print(f"Wrote {out_path}")


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Compute CLTS/MLTS linguistic complexity differences for each corpus."
    )
    parser.add_argument("--input-dir", default="input")
    parser.add_argument("--output-dir", default="linguistic features output")
    parser.add_argument("--output-files", default=None,
                        help="Comma-separated output file names (default: all in OUTPUT_PLAN).")
    parser.add_argument("--download-resources", action="store_true",
                        help="Download missing NLTK/spaCy resources (disabled by default).")
    args = parser.parse_args()

    output_files = ([n.strip() for n in args.output_files.split(",")]
                    if args.output_files else list(OUTPUT_PLAN))
    run(output_files, args.input_dir, args.output_dir, args.download_resources)


if __name__ == "__main__":
    main()