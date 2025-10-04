"""Lemmatize keywords"""

import os
import re

import nltk
import numpy as np
import pandas as pd
import unidecode
from common.tai_g_um_utils.custom_exceptions import PocType, TaiGUMError, UnhandledError
from nltk import pos_tag_sents, word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from .utils import clean_keywords

nltk_data_dir = os.getenv("nltk_data_dir", "/tmp/nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)

# Add the custom directory to NLTK's data path
if nltk_data_dir not in nltk.data.path:
    nltk.data.path.append(nltk_data_dir)

if not os.path.exists(os.path.join(nltk_data_dir, "corpora", "wordnet")):
    nltk.download("wordnet", download_dir=nltk_data_dir)

# Add the custom directory to NLTK's data path for punkt
if not os.path.exists(os.path.join(nltk_data_dir, "tokenizers", "punkt")):
    nltk.download("punkt", download_dir=nltk_data_dir)

# Add the custom directory to NLTK's data path for taggers
if not os.path.exists(os.path.join(nltk_data_dir, "taggers", "averaged_perceptron_tagger")):
    nltk.download("averaged_perceptron_tagger", download_dir=nltk_data_dir)


class Lemmatizer:
    """Lemmatise keywords"""

    # pylint: disable=dangerous-default-value

    def __init__(self) -> None:
        """Setup the base lemmatizer"""
        self.base_lemmatizer = WordNetLemmatizer()
        self.lemmatized_cache = {}

    def lemmatize(self, kw_list: list[str] = []) -> list[str]:
        """Lemmatize list of keywords

        Args:
            kw_list (list[str], optional): List of input keywords. Defaults to [].

        Returns:
            list[str]: Lemmatized keywords
        """
        try:
            self.lemmatize_add_new_words_to_cache(kw_list)
            out_list = [self.lemmatized_cache.get(kw, kw) for kw in kw_list]
            return out_list
        except TaiGUMError as e:
            e.message = f"Error in lemmatize: {e.message}"
            raise e
        except Exception as e:
            raise UnhandledError(f"Error in lemmatize: {e}", notify=True, poc_type=PocType.DS) from e

    def lemmatize_add_new_words_to_cache(self, kw_list: list[str] = []) -> None:
        """Lemmatize and save lemmatized versions into cache

        Args:
            kw_list (list[str], optional): List of input keywords. Defaults to [].
        """
        # pylint: disable=unnecessary-lambda
        try:
            df = pd.DataFrame({"raw_kw": kw_list})
            df = df[~df["raw_kw"].isin(self.lemmatized_cache)]

            df["basic_clean_kw"] = df["raw_kw"].map(clean_keywords)
            df["basic_clean_kw_in_cache"] = df["basic_clean_kw"].isin(self.lemmatized_cache)

            present_clean_words = df[df["basic_clean_kw_in_cache"]].set_index("raw_kw")["basic_clean_kw"].to_dict()
            for kw in present_clean_words:
                self.lemmatized_cache[kw] = self.lemmatized_cache[present_clean_words[kw]]

            df = df[~df["basic_clean_kw_in_cache"]]

            print(
                f"Lemmatizing {len(df)} new keywords, {len(kw_list)-len(df)} already found out of "
                f"{len(kw_list)}. Cache size {len(self.lemmatized_cache)}"
            )

            if len(df) == 0:
                return

            df["split_kw"] = df["basic_clean_kw"].map(lambda x: word_tokenize(x))  # type: ignore
            df["pos_tags"] = pos_tag_sents(df["split_kw"].to_list())
            lemmatize_inputs = np.concatenate([x for x in df["pos_tags"] if len(x) > 0]).tolist()
            lemmatized_keywords = self.lemmatize_keywords(lemmatize_inputs)
            df["lemm_kw"] = df["pos_tags"].map(lambda x: " ".join([lemmatized_keywords[kw[1]][kw[0]] for kw in x]))

            for row in df[["basic_clean_kw", "raw_kw", "lemm_kw"]].to_dict(orient="records"):
                self.lemmatized_cache[row["raw_kw"]] = row["lemm_kw"]
                self.lemmatized_cache[row["basic_clean_kw"]] = row["lemm_kw"]

        except TaiGUMError as e:
            e.message = f"Error in lemmatize_add_new_words_to_cache: {e.message}"
            raise e
        except Exception as e:
            raise UnhandledError(
                f"Error in lemmatize_add_new_words_to_cache: {e}",
                notify=True,
                poc_type=PocType.DS,
            ) from e

    def lemmatize_keywords(self, lemmatize_input_pairs: list) -> dict:
        """Lemmatize keywords POS combiantions

        Args:
            lemmatize_input_pairs (list): List of keywords,POS tag tuples

        Returns:
            dict: POS wise dict with POS as key and dict with keyword and lemmsatized version as key and values
        """
        # pylint: disable=consider-using-dict-items
        try:
            type_wise_keywords = {}
            for pair in lemmatize_input_pairs:
                if pair[1] not in type_wise_keywords:
                    type_wise_keywords[pair[1]] = set([pair[0]])
                else:
                    type_wise_keywords[pair[1]].add(pair[0])

            first_key_type_mapping = {"J": wordnet.ADJ, "V": wordnet.VERB, "N": wordnet.NOUN, "R": wordnet.ADV}

            type_wise_output = {}
            for pos_type in type_wise_keywords:
                tag = None
                if (isinstance(pos_type, str)) and (len(pos_type) >= 1):
                    tag = first_key_type_mapping.get(pos_type[0])

                if tag is not None:
                    type_wise_output[pos_type] = {
                        word: self.base_lemmatizer.lemmatize(word, tag) for word in type_wise_keywords[pos_type]
                    }
                else:
                    type_wise_output[pos_type] = {
                        word: self.base_lemmatizer.lemmatize(word) for word in type_wise_keywords[pos_type]
                    }
        except TaiGUMError as e:
            raise e
        except Exception as e:
            raise UnhandledError(f"Error in lemmatize_keywords: {e}", notify=True, poc_type=PocType.DS) from e

        return type_wise_output
