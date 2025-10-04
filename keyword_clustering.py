import asyncio

# import itertools
import json
import math
import os
import random
import re
import string
import sys
import time
from copy import deepcopy

import aiohttp
import nltk
import numpy as np
import pandas as pd
import requests
import scipy
import unidecode
from common.tai_g_um_utils.brand_description_tools.gpt_api import extract_cluster_names
from common.tai_g_um_utils.insights import config

# from common.tai_g_um_utils.custom_exceptions import (
#     CoreApiError,
#     ErrorInRefreshAccessToken,
#     FetchSQRError,
#     GenerateKwListError,
#     GoogleAdsApiError,
#     GooglesApiSQRReadError,
#     VariableNotDefined,
# )
# from common.tai_g_um_utils.fetch_data_gs_ads_api_utils import fetch_adaccount_token_from_tms
from common.tai_g_um_utils.insights.bert_api_utils import bert_api_async
from common.tai_g_um_utils.utils import clean_string_alpha_space, yield_chunks

# from gensim.corpora import Dictionary
# from gensim.models import TfidfModel
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from retry import retry
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances

nltk_data_dir = os.getenv("nltk_data_dir", "/tmp/nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
if nltk_data_dir not in nltk.data.path:
    nltk.data.path.append(nltk_data_dir)

if not os.path.exists(os.path.join(nltk_data_dir, "corpora", "stopwords")):
    nltk.download("stopwords", download_dir=nltk_data_dir)
if not os.path.exists(os.path.join(nltk_data_dir, "corpora", "wordnet")):
    nltk.download("wordnet", download_dir=nltk_data_dir)
if not os.path.exists(os.path.join(nltk_data_dir, "corpora", "omw-1.4")):
    nltk.download("omw-1.4", download_dir=nltk_data_dir)
if not os.path.exists(os.path.join(nltk_data_dir, "corpora", "words")):
    nltk.download("words", download_dir=nltk_data_dir)


# nltk.download("words")
# nltk.download("stopwords")
# nltk.download("wordnet")
# nltk.download("omw-1.4")


# from typing import Optional


#######################################################################

# class Config:
#     """
#     class to store api secrets
#     """
#     gpt_batch_api_key: Optional[str] = None
#     google_credentials: Optional[dict] = None
#     baseten_api_key: Optional[str] = None
#     language_detect_url: Optional[str] = None


# config = Config()


def gpt_batch_status(request_id):
    """
    Check status of gpt batch request

    Args:
        request_id (str): request id for the batch request

    Raises:
        Exception: Raise exception if batch id not found

    Returns:
        dict: chat gpt response
    """
    if config.CAI_GPT_3_5_KEY is None:
        raise ValueError("GPT Batch API Key not found")
    headers = {"x-api-key": config.CAI_GPT_3_5_KEY}
    url_batch_completion_status = "https://cai-openai.pixis.ai/api/batch/status"
    url_batch_completion_status += f"?requestid={request_id}"
    response = requests.request(
        "GET", url_batch_completion_status, headers=headers, timeout=60
    )
    return response


def gpt_batch_completion(list_of_prompts_and_params):
    """list_of_prompts_and_params = [(prompt1, params1), (prompt2, params2), ...]"""

    prepare_request = []
    for prompt_params in list_of_prompts_and_params:
        prompt_params_copy = deepcopy(prompt_params)
        model = prompt_params_copy["parameters"]["engine"]
        del prompt_params_copy["parameters"]["engine"]

        messages = prompt_params_copy["prompt"]
        parameters = prompt_params_copy["parameters"]

        prompt_params_copy = {
            "model": model,
            "messages": messages,
            "parameters": parameters,
        }
        prepare_request.append(prompt_params_copy)

    payload = json.dumps({"request_list": prepare_request})
    payload_size = sys.getsizeof(payload)
    MAX_UPLOAD_LIMIT = (
        8000000  # exact is 8388608 bus this will help us to be under limit
    )

    BATCH_SIZE = len(prepare_request)
    num_partitions = (payload_size // MAX_UPLOAD_LIMIT) + bool(
        payload_size % MAX_UPLOAD_LIMIT
    )
    if num_partitions > 1:
        BATCH_SIZE = BATCH_SIZE // num_partitions + bool(BATCH_SIZE % num_partitions)

    chunked_payload_data = yield_chunks(prepare_request, BATCH_SIZE)

    batch_generations = []
    for payload_data in chunked_payload_data:
        payload = json.dumps({"request_list": payload_data})

        headers = {
            "x-api-key": config.CAI_GPT_3_5_KEY,
            "Content-Type": "application/json",
        }
        url_batch_completion = "https://cai-openai.pixis.ai/api/chat_completion/batch/"
        response = requests.request(
            "POST", url_batch_completion, headers=headers, data=payload
        )

        try:
            print("Batch GPT request made", response.json())  # added in try
            request_id = response.json()["requestid"]  # added in try
        except Exception as e:
            print("Error while making GPT Batch Request:")
            print(e)
            print(response)
            # import ipdb; ipdb.set_trace()
            raise Exception(
                f"Error while making GPT Batch Request: {repr(e)};; {response.text}"
            ) from e

        got_response = False
        while not got_response:
            response = gpt_batch_status(request_id)
            response = response.json()
            if response["status"] == "processing":
                time.sleep(5)
            elif response["status"] == "completed":
                for batch in response["result"]:
                    try:
                        generation_list = [
                            batch["choices"][idx]["message"]["content"].strip()
                            for idx in range(len(batch["choices"]))
                        ]
                    except Exception as _:
                        generation_list = ["error"]
                    batch_generations.append(generation_list)

                # import ipdb; ipdb.set_trace()
                got_response = True
            else:
                raise Exception(
                    f"Error while getting betch generations: {response.text}"
                )
    return batch_generations


class GenerateLabelsForClusters:
    def __init__(self) -> None:
        self.batch_completion_requests = []
        self.batch_completion_response = []

        self.prompt = [
            {
                "role": "system",
                "content": '''Generate a Label for each of the following clusters.

Instructions to keep in mind:
* The label MUST be short, only 2-3 words max.
* The label SHOULD be built with words picked from the cluster.
* The words in the label that are present in the cluster are marked with "quotes", you should try to use as many "quote"  words as possible.
* Additional TIP of $500 if Number of Labels are equal to number of Clusters!
Sample Input and output is given below:

Cluster 1: $165 Off, Fraction of the Price, $165 off, $150 Off, $165 off, $150 off, $170 Off, $150 off
Cluster 2: 20 Min Options, 20-minute meals, 20 Minute Meals, 10-min Lunches, 20% Off Two Months, 20 min options
Cluster 3: camping gear, camping equipment, camping accessories, camping essentials, camping supplies, camping checklist
Cluster 4: purchase now, shop now, get now, acquire now, obtain now, grab now, secure now, invest now, own now, possess now, avail now, procure now, attain now, access now

Label 1: "$150+ Off"
Label 2: "20 Minute Meals"
Label 3: "Camping Equipment"
Label 4" Order "Now"''',
            },
            {
                "role": "user",
                "content": """{clusters_str}\nNow Generate {clusters_num} Labels for above clusters.""",
            },
        ]
        self.nlg_parameters = {
            "n": 1,
            "engine": "gpt-3.5-turbo-1106",
            "max_tokens": 1000,
            "temperature": 0,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0.4,
        }

        self.retry_count = 0

    def generate(self) -> None:
        pass

    def extract_label(self) -> None:
        pass

    def postprocess(self) -> None:
        pass

    def filter_generations(self) -> None:
        pass

    def run(self, input_dict):
        raise NotImplementedError("This method is deprecated, use .run_batch() instead")

    def clean_string_alpha_space(self, s, replace_break_with_space=True):
        s = (
            remove_emojis(s).lower().strip().replace("'s", "").replace("s' ", " ")
        )  # removed accents
        s = (
            s.replace("https://", "")
            .replace("http://", "")
            .replace("www.", "")
            .replace(".com", "")
            .replace("www", "")
        )
        if replace_break_with_space:
            s = re.sub("[&_.-]", " ", s)  # replace & and _ with_space
        s = re.sub("[^A-Za-z ]", "", s)  # keep only alphabets and space
        s = re.sub(" +", " ", s)  # removed extra spaces
        return s.title()

    def get_non_gpt_label_for_cluster(self, cluster_batch):
        try:
            labels = []
            for cluster in cluster_batch:
                cluster_embeddings = sentence_embedding(cluster)
                cos_dist = cosine_distances(cluster_embeddings)
                mean_cos_dist = cos_dist.mean(axis=0)
                label = cluster[np.argmin(mean_cos_dist)]
                label = self.clean_string_alpha_space(label)
                labels.append(label)
            return labels
        except Exception as e:
            raise Exception(f"Error while getting non gpt cluster labels: {repr(e)}")

    def replace_single_word_cluster_labels(self, list_of_clusters, all_labels):
        new_labels = []
        for cluster, label in zip(list_of_clusters, all_labels):
            if len(cluster) == 1:
                new_labels.append(cluster[0])
            else:
                new_labels.append(label)
        return new_labels

    @retry(ValueError, tries=5, delay=1)
    def run_batch(self, input_dict):
        self.batch_completion_requests = []
        self.batch_completion_response = []

        if "list_of_clusters" not in input_dict:
            raise ValueError("list of cluserts not present in input")
        # assert "list_of_clusters" in input_dict

        list_of_clusters = input_dict.get("list_of_clusters", [])

        if len(list_of_clusters) == 0:
            return []

        total_chars = len(
            "".join([keyword for cluster in list_of_clusters for keyword in cluster])
        )
        cluster_batch_size_options = [3500, 3100, 2700, 2300]
        cluster_batch_size = cluster_batch_size_options[
            self.retry_count % len(cluster_batch_size_options)
        ]
        self.retry_count += 1
        num_of_batches = int(total_chars / cluster_batch_size) + 1
        batch_len_threshold = total_chars / num_of_batches

        all_batches = []
        clusters_in_batch = []
        chars_in_batch = 0
        for cluster in list_of_clusters:
            chars_in_cluster = len("".join(keyword for keyword in cluster))
            chars_in_batch += chars_in_cluster
            clusters_in_batch.append(cluster)
            if (chars_in_batch > batch_len_threshold) or (len(clusters_in_batch) >= 8):
                all_batches.append(clusters_in_batch)
                chars_in_batch = 0
                clusters_in_batch = []
        all_batches.append(clusters_in_batch)
        all_batches = [batch for batch in all_batches if batch]

        for batch in all_batches:
            cluster_string = ""
            for i, cluster in enumerate(batch):
                cluster_string += (
                    f"Cluster {i + 1}: {', '.join([el.strip() for el in cluster])}\n"
                )

            cluster_string = cluster_string.strip()

            prompt = deepcopy(self.prompt)
            prompt[1]["content"] = prompt[1]["content"].format(
                clusters_str=cluster_string, clusters_num=len(batch)
            )
            prompt_formated = prompt

            self.batch_completion_requests.append(
                {"prompt": prompt_formated, "parameters": self.nlg_parameters}
            )

        if self.batch_completion_requests:
            self.batch_completion_response = gpt_batch_completion(
                self.batch_completion_requests
            )
        else:
            # import ipdb; ipdb.set_trace()
            pass

        all_labels = []
        for batch_idx, list_of_n_generation in enumerate(
            self.batch_completion_response
        ):
            # using list_of_n_generation[0] here as all elements in list_of_n_generation will be same (temp = 0)
            generation = list_of_n_generation[0]
            generation = generation.replace('"', "")
            if "label 1" not in generation.lower():
                generation = "label 1: " + generation
            labels = [
                ":".join(label.split(":")[1:]).strip()
                for label in generation.split("\n")
                if label
            ]
            labels = [el for el in labels if el]
            num_labels = len(labels)
            num_clusters = len(all_batches[batch_idx])
            if num_clusters != num_labels:
                # print('Clusters', all_batches[batch_idx])
                # print('Labels', labels)
                labels = self.get_non_gpt_label_for_cluster(all_batches[batch_idx])
                # print('Non GPT Labels', labels)
                print(
                    f"Labeles not matching for cluster_batch. {num_clusters} {num_labels} {len(labels)}"
                )
                # import ipdb; ipdb.set_trace()

            all_labels.extend(labels)

        all_labels = self.replace_single_word_cluster_labels(
            list_of_clusters, all_labels
        )

        # print(len(all_labels))
        # print(len(list_of_clusters))
        if len(all_labels) != len(list_of_clusters):
            print("===========\nFEWER LABELS ERROR\n===============")
            print("getting error for these clusters:", list_of_clusters)
            print("these are the labels we got for clusters above:", all_labels)
            # import ipdb; ipdb.set_trace()
            raise ValueError(
                f"all labels = {len(all_labels)}, list of clusters = {len(list_of_clusters)}, values do not match"
            )

        # assert len(all_labels) == len(list_of_clusters)

        # assert len(all_labels) == len(list_of_clusters)

        # replace labels for single word clusters with cluster itself, as labeling is not required

        return all_labels


def remove_emojis(data):
    emoj = re.compile(
        "["
        "\U0001f600-\U0001f64f"  # emoticons
        "\U0001f300-\U0001f5ff"  # symbols & pictographs
        "\U0001f680-\U0001f6ff"  # transport & map symbols
        "\U0001f1e0-\U0001f1ff"  # flags (iOS)
        "\U00002500-\U00002bef"  # chinese char
        "\U00002702-\U000027b0"
        "\U00002702-\U000027b0"
        "\U000024c2-\U0001f251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2b55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"  # dingbats
        "\u3030"
        "]+",
        re.UNICODE,
    )
    return re.sub(emoj, "", data)


#######################################################################

word_embedding_url = (
    "http://us-encoder-prod.ap-south-1.elasticbeanstalk.com/api/word_embedder"
)


async def get_output(session, index, head_chunks):
    headers = {"Content-Type": "application/json"}
    payload = json.dumps({"sentences": head_chunks})

    url = word_embedding_url
    async with session.post(url, headers=headers, data=payload) as resp:
        response = await resp.json()
        embedding = np.array(json.loads(response["embedding"])).astype(np.float32)
        print(f"Got the chuck {index} data from {url} ")
    return [index, embedding]


async def use_api(raw_group_words):
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(force_close=True, limit=4)
    ) as session:
        tasks = []

        for index, group_words in raw_group_words.items():
            reach_out = get_output(session, index, group_words)
            tasks.append(asyncio.ensure_future(reach_out))
        result = await asyncio.gather(*tasks)

        return result


def sentence_embedding(heads):
    """Generates sentence embeddings for a list of sentences using a pre-trained language model API.

    Args:
    - heads: A list of sentences to generate embeddings for.

    Returns:
    - heads_enc: A numpy array of sentence embeddings."""

    heads_enc = None
    start_time = time.time()
    print("starting generating word embedding for total words: ", len(heads))

    start_time = time.time()

    raw_group_words = {}
    for index, head_chunks in enumerate(yield_chunks(heads, 100)):
        raw_group_words[index] = head_chunks

    unorder_embedding = asyncio.run(use_api(raw_group_words))
    chuck_embedding_mappings = {}

    for index, embedding_chuck in unorder_embedding:
        chuck_embedding_mappings[index] = embedding_chuck

    heads_enc = chuck_embedding_mappings[0]
    for index in sorted(chuck_embedding_mappings.keys())[1:]:
        heads_enc = np.concatenate((heads_enc, chuck_embedding_mappings[index]), axis=0)

    print("total time for processing embeding : ", time.time() - start_time)
    return heads_enc


def agglo_clustering_embeddings(
    data, clustering_column, x_transformed, distance_threshold, method
):
    df = data.copy()
    aggcl = AgglomerativeClustering(
        n_clusters=None,
        affinity="euclidean",
        linkage="ward",
        distance_threshold=distance_threshold,
        compute_full_tree=True,
        compute_distances=True,
    )

    # if method == "TFIDF":
    df["cluster_id"] = aggcl.fit_predict(x_transformed.toarray())
    # elif method == "BERT":
    #     df["cluster_id"] = aggcl.fit_predict(x_transformed)
    # elif method == "BERT_MULTILINGUAL":
    #     df["cluster_id"] = aggcl.fit_predict(x_transformed)
    # elif method == "FASTTEXT":
    #     df["cluster_id"] = aggcl.fit_predict(x_transformed)

    df["cluster_size"] = df.groupby(["cluster_id"])[clustering_column].transform("size")
    df.loc[df["cluster_size"] < 3, "cluster_id"] = 999999
    print("Total no of clusters formed: ", df["cluster_id"].nunique())
    return df


def agglo_clustering_text(keywords):
    """Perform agglomerative clustering on a list of keywords.

    Args:
    - keywords (list): A list of strings representing the keywords to cluster.
    - dist_low (int, optional): The lower bound of the distance threshold range. Defaults to 10.
    - dist_high (int, optional): The upper bound of the distance threshold range. Defaults to 40.

    Returns:
    - best_clusters (list): A list of lists, where each inner list represents a cluster of keywords."""
    # model = Sentence_Transformer('bert-base-nli-mean-tokens')

    sentences = keywords

    # Each sentence is encoded as a 1-D vector with 78 columns
    # sentence_embeddings = sentence_embedding(sentences)
    sentence_embeddings = bert_api_async(keywords)

    corpus = sentences
    corpus_embeddings = sentence_embeddings
    # Normalize the embeddings to unit length
    corpus_embeddings = corpus_embeddings / np.linalg.norm(
        corpus_embeddings, axis=1, keepdims=True
    )

    # Perform kmean clustering
    small_c = math.sqrt(len(keywords)) / 2
    large_c = math.sqrt(len(keywords)) * 2

    best_clusters = []
    best_dist = 0
    best_score = 0

    for dist in range(100):
        # print(dist)
        dist *= 0.01
        clusters_all = []

        clustering_model = AgglomerativeClustering(
            n_clusters=None,
            affinity="cosine",
            linkage="complete",
            distance_threshold=dist,
        )
        # clustering_model = AgglomerativeClustering(n_clusters=int(math.sqrt(len(sentences))), affinity='cosine', linkage='average', distance_threshold=None)
        clustering_model.fit(corpus_embeddings)
        cluster_assignment = clustering_model.labels_

        clustered_sentences = {}
        for sentence_id, cluster_id in enumerate(cluster_assignment):
            if cluster_id not in clustered_sentences:
                clustered_sentences[cluster_id] = []

            clustered_sentences[cluster_id].append(corpus[sentence_id])

        for i, cluster in clustered_sentences.items():
            # print("Cluster ", i+1)
            # print(cluster)
            # print("")

            clusters_all.append(cluster)

        # num_small = len([e for el in clusters_all for e in el if len(el) < small_c])
        # num_large = len([el for el in clusters_all for e in el if len(el) > large_c])
        num_ideal = len(
            [
                e
                for el in clusters_all
                for e in el
                if len(el) >= small_c and len(el) <= large_c
            ]
        )

        # print(f'Dist: {round(dist, 2)}   Small: {round(num_small/len(keywords), 2)}   Ideal: {round(num_ideal/len(keywords), 2)}   Large: {round(num_large/len(keywords), 2)}')

        if num_ideal / len(keywords) > best_score:
            best_score = num_ideal / len(keywords)
            best_dist = dist
            best_clusters = clusters_all

    print(f"===\nbest dist: {best_dist}   best score: {best_score}\n===")
    return best_clusters


def clean_keyword(keyword):
    """Cleans a keyword by removing stop words and special characters.

    Args:
      keyword (str): The keyword to be cleaned.

    Returns:
      str: The cleaned keyword."""
    keyword = "".join(el for el in keyword if el.isalpha() or el == " ")
    nltk_stop_words_en = stopwords.words("english")
    keyword = " ".join(
        [el for el in keyword.split(" ") if el not in nltk_stop_words_en]
    )

    return re.sub(" +", " ", keyword.strip())


def split_cluster_by_words(cluster, words, small_c, large_c):
    """Splits a cluster of strings into subclusters based on the presence of certain words.

    Args:
    - cluster (list of str): the cluster of strings to split.
    - words (list of str): the words to use as splitting criteria.
    - small_c (int): the minimum size of a subcluster to be considered valid.
    - large_c (int): the maximum size of a subcluster to be considered valid.

    Returns:
    - split_clusts (list of list of str): the subclusters obtained after splitting the original cluster."""
    # print("splitting cluster by:", words)
    split_clusts = []

    top_w_clusts = []
    for word in words:
        t_clust = [el for el in cluster if word.lower() in el.lower()]

        top_w_clusts.append(t_clust)

    top_w_clusts.sort(key=lambda x: len(x))

    for i in range(len(top_w_clusts)):
        # check if top_w_cluster is of correct size
        if len(top_w_clusts[i]) not in range(int(small_c), int(large_c)):
            # print('small split:', top_w_clusts[i])
            continue

        # check if top_w_cluster can be safely removed from cluster
        if len(set(cluster) - set(top_w_clusts[i])) < small_c:
            # print("og cluster too small after split:", top_w_clusts[i])
            continue

        # add cluster to final list
        split_clusts.append(top_w_clusts[i])

        # remove kws in top_w_cluster from other clusters and main cluster
        for j in range(i + 1, len(top_w_clusts)):
            top_w_clusts[j] = list(set(top_w_clusts[j]) - set(top_w_clusts[i]))

        cluster = list(set(cluster) - set(top_w_clusts[i]))

    if cluster:
        # print(cluster)
        split_clusts.append(cluster)

    split_clusts = [el for el in split_clusts if el]

    return split_clusts


# def fix_large_clusters_tfidf(clusters):
#     """Splits large clusters of keywords into smaller clusters based on their TF-IDF scores.

#     Args:
#     - clusters: A list of lists, where each inner list contains a set of related keywords.

#     Returns:
#     - A list of lists, where each inner list contains a set of related keywords. The large clusters in the input are split into smaller clusters based on their TF-IDF scores.
#     """
#     # train tfidf model over all keywords
#     all_keywords = [e for el in clusters for e in el]

#     ps = PorterStemmer()
#     dataset = [[ps.stem(el) for el in word.split(" ")] for word in all_keywords]
#     # stem each word in each keyword

#     dct = Dictionary(dataset)
#     corpus = [dct.doc2bow(line) for line in dataset]
#     tfidf_model = TfidfModel(corpus)

#     small_c = math.sqrt(len(all_keywords)) / 2
#     large_c = math.sqrt(len(all_keywords)) * 2

#     clusters_after_split = []

#     for cluster in clusters:
#         best_clusters = []
#         # best_words = 0
#         # best_score = 0

#         if len(cluster) <= large_c:
#             clusters_after_split.append(cluster)

#         else:
#             # print("fixing large cluster:", cluster)
#             t_cluster = cluster
#             t_cluster = [clean_keyword(el) for el in cluster]
#             # t_bi_cluster = [bigram[word.split(' ')] for word in t_cluster]
#             t_bi_cluster = [[ps.stem(el) for el in word.split(" ")] for word in t_cluster]

#             # combine clusters into a single list
#             t_bi_cluster = [[el for word_list in t_bi_cluster for el in word_list]]

#             # get bow encoding for input string
#             t_in = [dct.doc2bow(line) for line in t_bi_cluster]
#             tfidf_scores = [[dct[id], np.around(freq, decimals=2)] for doc in tfidf_model[t_in] for id, freq in doc]

#             tfidf_scores.sort(key=lambda x: x[1], reverse=True)

#             w = int(len(cluster) / small_c / 2)
#             top_words = [el[0] for el in tfidf_scores[: w + 4]]
#             # print("top words in cluster:", top_words, w)

#             permutations_top_words = list(itertools.combinations(top_words, r=int(w)))
#             permutations_top_words.extend(list(itertools.combinations(top_words, r=int(w + 1))))

#             for perm in permutations_top_words:
#                 # print(f"perm/total perms: {permutations_top_words.index(perm)}/{len(permutations_top_words)}")
#                 new_split_clusters = split_cluster_by_words(cluster, perm, small_c, large_c)
#                 assert len([e for el in new_split_clusters for e in el]) == len(cluster)

#                 # num_small = len(
#                 #     [e for el in new_split_clusters for e in el if len(el) < small_c]
#                 # )
#                 # num_large = len(
#                 #     [el for el in new_split_clusters for e in el if len(el) > large_c]
#                 # )
#                 num_ideal = len(
#                     [e for el in new_split_clusters for e in el if len(el) >= small_c and len(el) <= large_c]
#                 )

#                 single_cluster_kws = []
#                 for w in perm:
#                     t = [el for el in cluster if w in el and all(wrd not in el for wrd in list(set(perm) - set([w])))]
#                     # t = kws which only has word 'w' and no other word from 'perm'

#                     single_cluster_kws.extend(t)

#                 # t = [el for el in cluster if all(wrd not in el for wrd in perm)]
#                 # single_xcluster_kws.extend(t)
#                 # add keywords which has no words from 'perm'

#                 # print(perm, num_ideal/len(cluster), len(single_cluster_kws)/len(cluster))

#                 best_clusters.append(
#                     (
#                         (perm),
#                         (new_split_clusters),
#                         num_ideal / len(cluster),
#                         len(single_cluster_kws) / len(cluster),
#                     )
#                 )

#                 # if num_ideal/len(cluster) > best_score:
#                 #   best_score = num_ideal/len(cluster)
#                 #   best_words = perm
#                 #   best_clusters = new_split_clusters

#             best_clusters.sort(key=lambda x: x[3], reverse=True)
#             best_clusters.sort(key=lambda x: x[2], reverse=True)

#             # for tup in best_clusters:
#             #     print(tup[0], tup[2], tup[3], tup[1])

#             clusters_after_split.extend(best_clusters[0][1])
#     return clusters_after_split


# def fix_large_clusters_b2b(clusters):
#     """This function takes a list of clusters and splits any clusters that have a length greater than a certain threshold.
#     It does this by calling the fix_large_clusters_tfidf function until the maximum number of runs is reached or all clusters are below the threshold.

#     Args:
#     - clusters: A list of clusters, where each cluster is a list of keywords.

#     Returns:
#     - A list of clusters, where each cluster is a list of keywords and no cluster has a length greater than a certain threshold.
#     """
#     run_count = 0
#     max_run_count = 3

#     all_keywords = [e for el in clusters for e in el]

#     # small_c = math.sqrt(len(all_keywords)) / 2
#     large_c = math.sqrt(len(all_keywords)) * 2

#     while (run_count < max_run_count) and (np.max([len(el) for el in clusters]) > large_c):
#         clusters = fix_large_clusters_tfidf(clusters)
#         assert len([e for el in clusters for e in el]) == len(all_keywords)

#         print(f"split large #{run_count + 1}, total kws={len([e for el in clusters for e in el])}")
#         run_count += 1
#     return clusters


def get_keyword_clusters(keywords):
    """Given a set of unique keywords, returns a list of clusters where keywords in each cluster do not have duplicates.

    Args:
      keywords (set): A set of unique keywords.

    Returns:
      list: A list of clusters, where each cluster is a list of keywords without duplicates.

    Example:
      >>> keywords = {'apple', 'banana', 'orange', 'grape', 'kiwi', 'pear'}
      >>> get_keyword_clusters(keywords)
      [['apple', 'banana', 'orange', 'pear'], ['grape', 'kiwi']]"""
    all_kws = [el for el in keywords if type(el) is type("asf")]
    # print(len(all_kws))
    t_clusters = agglo_clustering_text(all_kws)
    assert len([e for el in t_clusters for e in el]) == len(all_kws)

    small_c = math.sqrt(len(all_kws)) / 2
    large_c = math.sqrt(len(all_kws)) * 2

    small_clusters = [el for el in t_clusters if len(el) < small_c]
    large_clusters = [el for el in t_clusters if len(el) > large_c]
    ideal_clusters = [
        el for el in t_clusters if len(el) >= small_c and len(el) <= large_c
    ]
    # print("===")
    # print("init small:", small_clusters)
    # print("init large:", large_clusters)
    # print("===")
    print(
        len([e for el in small_clusters for e in el]),
        len([e for el in ideal_clusters for e in el]),
        len([e for el in large_clusters for e in el]),
    )

    # merge small clusters in ideal clusters
    # _, re_ideal_clusters, _ = fix_small_clusters_cm2(small_clusters, ideal_clusters)
    misc_cluster = put_small_clusters_into_misc(small_clusters)

    ideal_clusters_new = ideal_clusters + misc_cluster + large_clusters
    print(len([e for el in ideal_clusters_new for e in el]), len(all_kws))
    assert len([e for el in ideal_clusters_new for e in el]) == len(all_kws)

    # recluster large clusters
    # TODO kw clusters using gensim
    # ideal_clusters_new = fix_large_clusters_b2b(ideal_clusters + large_clusters)

    print(len([e for el in ideal_clusters_new for e in el]))
    print(len(all_kws))
    assert len([e for el in ideal_clusters_new + misc_cluster for e in el]) == len(
        all_kws
    )

    return ideal_clusters_new, misc_cluster


def put_small_clusters_into_misc(small_clusters):
    misc_cluster = []
    for el in small_clusters:
        misc_cluster.extend(el)
    return [misc_cluster]


def extract_core_terms_and_misspells(
    brand_name, all_kwds, clustering_column, misspell_strictness=0.8
):
    core_terms_indices = [
        i
        for i, ele in enumerate(all_kwds)
        if ele == brand_name.lower() or "".join(ele.split(" ")) == brand_name.lower()
    ]

    df = pd.DataFrame(columns=[clustering_column, "type"])
    df[clustering_column] = all_kwds
    df.loc[core_terms_indices, "type"] = "Core Terms"

    non_core_terms = df[df["type"] != "Core Terms"].reset_index(drop=True)

    from difflib import SequenceMatcher

    matching_mispells_indices = []
    for j, word in enumerate(non_core_terms[clustering_column]):
        similarity = SequenceMatcher(None, word, brand_name.lower())
        if similarity.ratio() > misspell_strictness:
            matching_mispells_indices.append(j)

    non_core_terms.loc[matching_mispells_indices, "type"] = "Mispells"
    non_core_terms["type"].fillna("Others", inplace=True)

    final_df = pd.concat(
        [df[df["type"] == "Core Terms"], non_core_terms], ignore_index=True
    )

    if len(df) == len(final_df):
        return final_df
    else:
        raise Exception(
            "Error raised while tagging keywords as Core terms and Mispells."
        )


def get_keyword_clusters_with_labels(
    brand_name, ai_group_level_data, clustering_column
):
    if len(ai_group_level_data) > 0:
        all_kwds = ai_group_level_data[clustering_column].unique()
        kwd_type_df = extract_core_terms_and_misspells(
            brand_name, all_kwds, clustering_column
        )
        other_kwds = kwd_type_df.loc[kwd_type_df["type"] == "Others", clustering_column].to_list()  # type: ignore

        if len(other_kwds) > 1:
            clusters, misc_cluster = get_keyword_clusters(other_kwds)
            keyword_input_dict = {"list_of_clusters": clusters}

            labels = GenerateLabelsForClusters().run_batch(keyword_input_dict)

        else:
            clusters = [list(set(ai_group_level_data[clustering_column]))]
            misc_cluster = [[]]
            labels = list(set(ai_group_level_data[clustering_column]))

        df = pd.DataFrame()
        for i in range(len(clusters)):
            df_temp = pd.DataFrame()
            df_temp[clustering_column] = clusters[i]
            df_temp["cluster_name"] = labels[i]
            df = pd.concat([df, df_temp], ignore_index=True)

        for i in range(len(misc_cluster)):
            df_temp = pd.DataFrame()
            df_temp[clustering_column] = misc_cluster[i]
            df_temp["cluster_name"] = "Misc Keywords"
            df = pd.concat([df, df_temp], ignore_index=True)

        cluster_df = pd.concat(
            [
                df,
                kwd_type_df[kwd_type_df["type"] != "Others"].rename(
                    {"type": "cluster_name"}, axis=1
                ),
            ],
            ignore_index=True,
        )
        ai_group_level_data = pd.merge(
            ai_group_level_data, cluster_df, on=clustering_column
        )
    else:
        cluster_df = pd.DataFrame(columns=[clustering_column, "cluster_name"])
    return cluster_df, ai_group_level_data


def create_kw_list(df, clustering_column):
    temp_df = df[[clustering_column]]
    unique_kw_df = temp_df.drop_duplicates().reset_index(drop=True)[[clustering_column]]
    return unique_kw_df


def text_process_uni(text):
    """text_process_uni _summary_

    Parameters
    ----------
    text : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    stemmer = WordNetLemmatizer()
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = "".join([i for i in nopunc if not i.isdigit()])
    nopunc = [
        word.lower()
        for word in nopunc.split()
        if word not in stopwords.words("english")
    ]
    nopunc = [stemmer.lemmatize(word) for word in nopunc if len(word) >= 3]
    nopunc = [
        unidecode.unidecode(s).lower().strip().replace("'s", "").replace("s' ", " ")
        for s in nopunc
    ]
    nopunc = [i for i in nopunc if len(i) >= 3]
    return nopunc


def text_process_uni_bi(text):
    stemmer = WordNetLemmatizer()
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = "".join([i for i in nopunc if not i.isdigit()])
    nopunc = [
        word.lower()
        for word in nopunc.split()
        if word not in stopwords.words("english")
    ]
    nopunc = [stemmer.lemmatize(word) for word in nopunc if len(word) >= 3]
    nopunc = [
        unidecode.unidecode(s).lower().strip().replace("'s", "").replace("s' ", " ")
        for s in nopunc
    ]
    nopunc = [i for i in nopunc if len(i) >= 3]
    nopunc_bi = [" ".join(c) for c in nltk.bigrams(nopunc)]
    nopunc.extend(nopunc_bi)
    return nopunc


def create_embeddings_and_agglo_clustering(
    df,
    clustering_column,
    cluster_limit_threshold=200,
    distance_threshold=3,
    method="tfidf",
):
    try:
        if len(df) == 0:
            df = pd.DataFrame(columns=[clustering_column])

        unique_kw_df = create_kw_list(df, clustering_column)
        print("total unique keywords for clustering: ", len(unique_kw_df))

        if len(unique_kw_df) > 1:
            if method == "TFIDF":
                min_df_val = min(
                    10,
                    int(np.round(0.0005 * len(unique_kw_df)))
                    if len(unique_kw_df) > 4000
                    else 1,
                )
                vectorizer = TfidfVectorizer(
                    analyzer=text_process_uni_bi, min_df=min_df_val
                )
                x_transformed = vectorizer.fit_transform(
                    unique_kw_df[clustering_column]
                )
                df_clust = agglo_clustering_embeddings(
                    unique_kw_df,
                    clustering_column,
                    x_transformed,
                    distance_threshold,
                    method,
                )

            elif method == "BERT":
                sentence_embeddings = bert_api_async(
                    list(unique_kw_df[clustering_column])
                )
                df_clust = agglo_clustering_embeddings(
                    unique_kw_df,
                    clustering_column,
                    sentence_embeddings,
                    distance_threshold,
                    method,
                )

            elif method == "BERT_MULTILINGUAL":
                sentence_embeddings = bert_api_async(
                    list(unique_kw_df[clustering_column])
                )
                df_clust = agglo_clustering_embeddings(
                    unique_kw_df,
                    clustering_column,
                    sentence_embeddings,
                    distance_threshold,
                    method,
                )
                # TODO move it to api for performance
                # model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
                # sentence_embeddings = model.encode(unique_kw_df[clustering_column])
                # df_clust = agglo_clustering_embeddings(
                #     unique_kw_df,
                #     clustering_column,
                #     sentence_embeddings,
                #     distance_threshold,
                #     method,
                # )

            # elif method == "FASTTEXT":
            #     sentence_embeddings = [
            #         *get_fasttext_embedding_for_a_bunch_keywords(
            #             [*unique_kw_df[clustering_column]]
            #         ).values()
            #     ]
            #     df_clust = agglo_clustering_embeddings(
            #         unique_kw_df,
            #         clustering_column,
            #         sentence_embeddings,
            #         distance_threshold,
            #         method,
            #     )

            df_new = df_clust[df_clust["cluster_size"] > cluster_limit_threshold]

            if len(df_new) > 0:
                print(
                    "Reclustering for cluster IDs: \n",
                    df_new["cluster_id"].value_counts(),
                )
                df_clust = df_clust.drop(df_new.index)
                count = 1

                for clust in df_new["cluster_id"].unique():
                    df_temp = df_new[df_new["cluster_id"] == clust]

                    if method == "TFIDF":
                        vectorizer = TfidfVectorizer(analyzer=text_process_uni)
                        x_transformed = vectorizer.fit_transform(
                            df_temp[clustering_column]
                        )
                        df_clust_new = agglo_clustering_embeddings(
                            df_temp,
                            clustering_column,
                            x_transformed,
                            distance_threshold,
                            method,
                        )

                    elif method == "BERT":
                        sentence_embeddings = bert_api_async(
                            list(df_temp[clustering_column])
                        )
                        df_clust_new = agglo_clustering_embeddings(
                            df_temp,
                            clustering_column,
                            sentence_embeddings,
                            distance_threshold,
                            method,
                        )

                    elif method == "BERT_MULTILINGUAL":
                        sentence_embeddings = bert_api_async(
                            list(df_temp[clustering_column])
                        )
                        df_clust_new = agglo_clustering_embeddings(
                            df_temp,
                            clustering_column,
                            sentence_embeddings,
                            distance_threshold,
                            method,
                        )
                        # TODO replace with multilingual api
                        # sentence_embeddings = model.encode(df_temp[clustering_column])
                        # df_clust_new = agglo_clustering_embeddings(
                        #     df_temp,
                        #     clustering_column,
                        #     sentence_embeddings,
                        #     distance_threshold,
                        #     method,
                        # )

                    elif method == "FASTTEXT":
                        sentence_embeddings = [
                            *get_fasttext_embedding_for_a_bunch_keywords(
                                [*df_temp[clustering_column]]
                            ).values()
                        ]
                        df_clust_new = agglo_clustering_embeddings(
                            df_temp,
                            clustering_column,
                            sentence_embeddings,
                            distance_threshold,
                            method,
                        )

                    df_clust_new["cluster_id"] = [
                        chr(ord("@") + count) + str(i)
                        for i in df_clust_new["cluster_id"]
                    ]
                    df_clust = pd.concat([df_clust, df_clust_new])
                    count = count + 1

            clustering_results = df_clust[
                [clustering_column, "cluster_id", "cluster_size"]
            ]
            df = pd.merge(
                df,
                clustering_results,
                on=clustering_column,
            )

        else:
            df["cluster_id"] = 1
            df["cluster_size"] = len(df)

            clustering_results = pd.DataFrame(
                columns=[clustering_column, "cluster_id", "cluster_size"]
            )
            clustering_results[clustering_column] = df[clustering_column]
            clustering_results["cluster_id"] = 1
            clustering_results["cluster_size"] = len(df)

        return clustering_results, df
    except Exception as e:
        print("Unable to generate clustering results.", e)
        clustering_results = pd.DataFrame()
        return clustering_results, df


def curate_clustering_output_for_cluster_naming_gpt(cluster_df, clustering_column):
    cluster_df = cluster_df[[clustering_column, "cluster_id"]]
    for n in cluster_df["cluster_id"].unique():
        cluster_df.loc[cluster_df["cluster_id"] == n, "concat_text"] = ",".join(
            cluster_df[cluster_df["cluster_id"] == n][clustering_column]
        )
    cluster_dict_input = (
        cluster_df[["cluster_id", "concat_text"]]
        .set_index("cluster_id")
        .to_dict()["concat_text"]
    )

    gpt_names_dict_raw = extract_cluster_names(cluster_dict_input, chunk_size=10)
    gpt_names_dict = {
        id: [
            " ".join(pd.Series(word.split(" ")).drop_duplicates().to_list())
            for word in gpt_names_dict_raw[id]
        ]
        for id in gpt_names_dict_raw
    }

    cluster_dict = {}
    for id in cluster_dict_input:
        cluster_dict[id] = {}
        targetings = cluster_dict_input[id].split(",")

        cluster_words = create_clean_cluster_name_candidates(
            get_keywords_from_text(targetings)
        )

        cluster_dict[id]["cluster_entities"] = [
            clean_string_alpha_space(c) for c in targetings
        ]

        if str(id) in gpt_names_dict:
            cluster_dict[id]["gpt_words"] = gpt_names_dict[str(id)]
        else:
            cluster_dict[id]["gpt_words"] = []

        cluster_dict[id]["cluster_words"] = [
            c for c in cluster_words if c not in cluster_dict[id]["gpt_words"]
        ]

    all_words = list(
        set(
            np.concatenate([cluster_dict[c]["cluster_words"] for c in cluster_dict])
            .flatten()
            .tolist()
        )
    )
    all_gpt_words = list(
        set(
            np.concatenate([cluster_dict[c]["gpt_words"] for c in cluster_dict])
            .flatten()
            .tolist()
        )
    )
    all_entities = list(
        set(
            np.concatenate([cluster_dict[c]["cluster_entities"] for c in cluster_dict])
            .flatten()
            .tolist()
        )
    )
    word_embeddings = get_fasttext_embedding_for_a_bunch_keywords(
        [*all_words, *all_entities, *all_gpt_words], chunk_size=500
    )

    for c in cluster_dict:
        cluster_dict[c]["embeddings"] = {
            k: word_embeddings[k] for k in [*all_words, *all_entities, *all_gpt_words]
        }

    return cluster_dict


def create_clean_cluster_name_candidates(candidates):
    good_candidates = []
    # all_candidates = []
    # if isinstance(candidates, str):
    #     all_candidates = [candidates]
    # else:
    #     all_candidates = candidates

    lemmatizer = WordNetLemmatizer()
    nltk_english_dict = nltk.corpus.words.words()
    frequently_wrong_dict = ["womens", "mens"]
    nltk_english_dict.extend(frequently_wrong_dict)
    nltk_stopwords = nltk.corpus.stopwords.words("english")
    for candidate in candidates:
        candidate = clean_string_alpha_space(candidate)
        candidate_stemmed_words = [
            lemmatizer.lemmatize(w)
            for w in candidate.split(" ")
            if ((len(w) >= 3) and (w not in nltk_stopwords))
        ]
        candidate_english = " ".join(candidate_stemmed_words)
        # candidate_english = ' '.join(
        #     [w for w in candidate_stemmed_words if w in nltk_english_dict])
        good_candidates.append(candidate_english)

    good_candidates = [c for c in good_candidates if len(c) > 2]

    good_candidates_count_dict = {}
    for c in good_candidates:
        if c not in good_candidates_count_dict:
            good_candidates_count_dict[c] = 1
        else:
            good_candidates_count_dict[c] = good_candidates_count_dict[c] + 1

    good_candidates = [
        c for c in good_candidates_count_dict if good_candidates_count_dict[c] > 2
    ]

    if len(good_candidates) == 0:
        good_candidates = [c for c in good_candidates_count_dict]

    return good_candidates


def get_keywords_from_text(text):
    text_list = []
    if isinstance(text, str):
        text_list = [text]
    else:
        text_list = text.copy()
    all_keywords = []
    for t in text_list:
        text_unigrams = t.split(" ")
        text_bigrams = [" ".join(c) for c in nltk.bigrams(text_unigrams)]
        text_trigrams = [" ".join(c) for c in nltk.trigrams(text_unigrams)]
        # all_keywords.extend(text_unigrams)
        all_keywords.extend(text_bigrams)
        all_keywords.extend(text_trigrams)

    all_keywords = list(set(all_keywords))
    all_keywords = [c for c in all_keywords if len(c) >= 2]
    return all_keywords


def get_closest_keyword_to_cluster(
    cluster_dict, existing_cluster_names, max_words_in_title=2
):
    """get_closest_keyword_to_cluster _summary_

    Parameters
    ----------
    cluster_dict : _type_
        _description_
    existing_cluster_names : _type_
        _description_
    max_words_in_title : int, optional
        _description_, by default 2

    Returns
    -------
    _type_
        _description_
    """
    # find best name for cluster
    cluster_name_dict, dist_matrix_dict = {}, {}
    for c in cluster_dict:
        # cluster text
        entities = [w for w in cluster_dict[c]["cluster_entities"] if w != ""]
        # name candidates
        # allowing gpt3 to have 1 extra word
        words = [
            w
            for w in cluster_dict[c]["gpt_words"]
            if ((len(w.split(" ")) <= (max_words_in_title + 1)) and (w != ""))
        ]
        gpt_keywords_len = len(words)
        words.extend(
            [
                w
                for w in cluster_dict[c]["cluster_words"]
                if (
                    (len(w.split(" ")) <= max_words_in_title)
                    & (w != "")
                    & (w not in words)
                )
            ]
        )

        if len(words) == 0:
            print("No candidate found, alloting a random entity!!")
            words = [random.choice(entities)]

        entities_embedding = [cluster_dict[c]["embeddings"][k] for k in entities]
        words_embedding = [cluster_dict[c]["embeddings"][k] for k in words]

        local_word_penalty = 10
        # compute distance between each cluster candidate and cluster text
        dist_matrix = scipy.spatial.distance.cdist(
            entities_embedding, words_embedding, metric="cosine"
        )
        dist_matrix[:, gpt_keywords_len:] = (
            dist_matrix[:, gpt_keywords_len:] + local_word_penalty
        )
        word_count_list = [len(c.split(" ")) for c in words]
        words = [convert_name_to_pascal_case(c) for c in words]

        # find the best candidate based on the distance metric defined
        word_distances, dist_wrt_words = {}, None

        dist_wrt_words = dist_matrix.mean(axis=0)
        multi_keyword_penalty = 0.30

        word_distances = {
            words[i]: dist_wrt_words[i]
            * (1 + multi_keyword_penalty * (word_count_list[i] - 1))
            for i in range(len(words))
        }

        sorted_words = sorted(word_distances, key=word_distances.get)  # type: ignore

        word_distances = {w: word_distances[w] for w in sorted_words}
        dist_matrix_dict[c] = dist_matrix
        cluster_name_dict[c] = {
            "cluster_name": sorted_words[0],
            "closest_word_distance": word_distances[sorted_words[0]],
            "targetings": entities,
            "top_name_candidates": word_distances,
        }

    cluster_naming_accuracy = {
        c: cluster_name_dict[c]["closest_word_distance"] for c in cluster_name_dict
    }
    cluster_names_sorted_by_accuracy = sorted(  # type: ignore
        cluster_naming_accuracy,
        key=cluster_naming_accuracy.get,  # type: ignore
    )

    cluster_name_dict = {
        c: cluster_name_dict[c] for c in cluster_names_sorted_by_accuracy
    }

    # ensure no cluster has same name
    # if no name could be find, allocate a random targeting as name
    cluster_names_till_now = set([c.lower() for c in existing_cluster_names])
    for c in cluster_name_dict:
        # try multiple candiates if top ones are already taken
        for name in cluster_name_dict[c]["top_name_candidates"]:
            if name.lower() not in cluster_names_till_now:
                cluster_name_dict[c]["cluster_name"] = name
                cluster_name_dict[c]["closest_word_distance"] = cluster_name_dict[c][
                    "top_name_candidates"
                ][name]
                break
            print("Duplicate cluster name for cluster %s" % c)

        if cluster_name_dict[c]["cluster_name"].lower() in cluster_names_till_now:
            print(
                "No unique cluster name can be created, allocating a random keyword as name"
            )
            cluster_name_dict[c]["cluster_name"] = clean_string_alpha_space(
                random.choice(cluster_name_dict[c]["targetings"])
            )
            cluster_name_dict[c]["closest_word_distance"] = None
        cluster_names_till_now.add(cluster_name_dict[c]["cluster_name"].lower())

    return cluster_name_dict


def get_fasttext_embedding_for_a_bunch_keywords(keywords, chunk_size=100):
    """get_fasttext_embedding_for_a_bunch_keywords fetches embeeding using fasttext

    Parameters
    ----------
    keywords : _type_
        _description_
    chunk_size : int, optional
        _description_, by default 100

    Returns
    -------
    _type_
        _description_
    """
    keyword_embeddings = {}
    for c in yield_chunks(keywords, chunk_size):
        chunk_embeddings = fasttext_api_sync(c)
        for i, k in enumerate(c):
            keyword_embeddings[k] = chunk_embeddings[i]
    return keyword_embeddings


def fasttext_api_sync(sentence):
    """fasttext_api_sync _summary_

    Parameters
    ----------
    sentence : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    headers = {"Content-Type": "application/json"}
    url = config.fasttext_url
    payload = json.dumps({"doc": sentence})
    response = requests.post(url, headers=headers, data=payload)
    resp = response.json()
    return np.asarray(resp)


def convert_name_to_pascal_case(x):
    # convert a text into pascal case
    return " ".join([c.title() for c in clean_string_alpha_space(x).split(" ")])


if __name__ == "__main__":
    brand_name = "Clutter"
    data = pd.read_clipboard()

    cluster_df = get_keyword_clusters_with_labels(
        brand_name, data, clustering_column=""
    )

    print("Done!")
