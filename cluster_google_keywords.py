import time
import numpy as np
import pandas as pd
import warnings
import random
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from itertools import chain
from sklearn.cluster import AgglomerativeClustering
from .config import ST_BERT_CONFIG, MAX_KEYWORDS_CUSTOM_ADGS, UNGROUPED_CLUSTER_SUFFIX
from .fetch_data_st_bert_model_api import bert_api_async
from .gpt3_cluster_naming import extract_cluster_names
from .lemmatizer import Lemmatizer
from .utils import chunks, clean_keywords, group_sim_from_matrix, compute_mean, group_sim_from_matrix_percentile


warnings.filterwarnings("ignore")


class ClusteringError(Exception):
    """Cluster exception handler"""

    def __init__(self, message=None):
        self.message = message
        super().__init__(self.message)


def cluster_keywords(keywords: list[str], lemmatizer: Lemmatizer, relevance_score_dict: dict[str, float] | None = None):

    if relevance_score_dict is None:
        relevance_score_dict = {}

    out_cols = ["text", "cluster_name", "similar_clusters", "cluster_score"]

    if len(keywords) > 3:
        clustering_obj = ClusterKeywords(
            keywords=keywords, lemmatizer=lemmatizer, relevance_score_dict=relevance_score_dict
        )
        cluster_df = clustering_obj.cluster_df
        keywords_df = cluster_df.rename(columns={"keywords": "text", "relevance_score": "cluster_score"})[
            ["text", "cluster_name", "similar_clusters", "cluster_score"]
        ].explode("text")
    else:
        keywords_df = pd.DataFrame({"text": [], "cluster_name": [], "similar_clusters": [], "cluster_score": []})

    missing_keywords = list(set(keywords) - set(keywords_df["text"].to_list()))
    missing_keywords_df = pd.DataFrame(
        {
            "text": missing_keywords,
            "cluster_name": [f"{UNGROUPED_CLUSTER_SUFFIX}{i+1}" for i in range(len(missing_keywords))],
            "similar_clusters": [[] for _ in missing_keywords],
            "cluster_score": [relevance_score_dict.get(keyword) for keyword in missing_keywords],
        }
    )

    keywords_df = pd.concat([keywords_df, missing_keywords_df], axis=0)
    keywords_df = keywords_df[out_cols]

    return keywords_df


class ClusterKeywords:
    max_keywords_for_clustering = MAX_KEYWORDS_CUSTOM_ADGS
    similarity_cutoff = 0.65

    def __init__(
        self,
        keywords: list[str],
        lemmatizer: Lemmatizer,
        relevance_score_dict: dict[str, float] | None = None,
        recm_clus_size=5,
        min_clus_size=3,
        distance_step=0.05,
        clustering_distance_cutoff=0.35,
        subphrases_sim_weight=0.6,
    ) -> None:
        self.lemmatizer = lemmatizer
        self.recm_clus_size = recm_clus_size
        self.min_clus_size = min_clus_size
        self.distance_step = distance_step
        self.subphrases_sim_weight = subphrases_sim_weight
        self.clustering_distance_cutoff = clustering_distance_cutoff

        if relevance_score_dict is None:
            relevance_score_dict = {}
        self.relevance_score_dict = relevance_score_dict

        self.keywords = keywords[: ClusterKeywords.max_keywords_for_clustering]
        self.raw_keywords_df = pd.DataFrame({"keyword": keywords})
        self.raw_keywords_df["relevance_score"] = self.raw_keywords_df["keyword"].map(
            lambda x: self.relevance_score_dict.get(x)
        )
        self.preprocess()
        self.create_clusters()
        self.post_processing()

    def preprocess(self) -> None:
        t_init = time.time()

        self.clean_entities()
        self.create_sub_phrases()
        self.create_keyword_ids()
        self.compute_phrases_similarity()
        print("Time taken to preprocess : ", time.time() - t_init)

    def create_clusters(self):
        t_init = time.time()
        comprehensive_clustering_obj = Comprehensive_Clustering(
            tar_sim_matrix=self.sim_matrix,
            tar_id_order=[self.keyword_name_id_link[tar] for tar in self.matrix_kw_order],
            recm_clus_size=self.recm_clus_size,
            min_clus_size=self.min_clus_size,
            distance_step=self.distance_step,
            clustering_distance_cutoff=self.clustering_distance_cutoff,
            name_dict=self.keyword_id_name_link,
        )
        keyword_idx_dict = {kw: i for i, kw in enumerate(self.matrix_kw_order)}

        scored_targetings_df = comprehensive_clustering_obj.targetings
        scored_targetings_df["cluster_id"] = scored_targetings_df["cohort_id"]
        scored_targetings_df["keyword_name"] = scored_targetings_df["id"].map(lambda x: self.keyword_id_name_link[x])

        scored_targetings_df["idx"] = scored_targetings_df["keyword_name"].map(lambda x: keyword_idx_dict[x])
        self.scored_targetings_df = self.raw_keywords_df.merge(scored_targetings_df, on="keyword_name", how="inner")
        print("Time taken to create clusters : ", time.time() - t_init)

    def post_processing(self):
        t_init = time.time()
        cluster_df = self.group_cluster(self.scored_targetings_df)
        print("Time taken to post process clusters : ", time.time() - t_init)
        cluster_names = self.fetch_cluster_names(cluster_df)

        pascal_case_cluster_names = {
            cluster_id: "".join([c.title() for c in cluster_name.split(" ")])
            for cluster_id, cluster_name in cluster_names.items()
        }
        cluster_df["cluster_name"] = cluster_df["cluster_id"].map(lambda x: pascal_case_cluster_names[x])
        cluster_df["similar_clusters"] = cluster_df["similar_clusters"].map(
            lambda x: [{"cluster_name": pascal_case_cluster_names[k], "similarity": x[k]} for k in x]
        )
        self.cluster_df = cluster_df

        print("Time taken to post process clusters : ", time.time() - t_init)

    def fetch_cluster_names(self, cluster_df: pd.DataFrame) -> dict:
        cluster_names = {}
        cluster_entity_dict = cluster_df.set_index("cluster_id")["keywords"].to_dict()

        gpt_cluster_names = extract_cluster_names(cluster_entity_dict, chunk_size=10)

        scored_gpt_cluster_names = {}
        # score cluster name if sufficient clusters
        if len(gpt_cluster_names) >= 2:
            scored_gpt_cluster_names = self.compute_cluster_name_relevance(
                cluster_name_dict=gpt_cluster_names, cluster_entity_dict=cluster_entity_dict
            )
            for clus in scored_gpt_cluster_names:
                scored_gpt_cluster_names[clus] = sorted(
                    scored_gpt_cluster_names[clus], key=scored_gpt_cluster_names[clus].get, reverse=True
                )
        else:
            for clus in gpt_cluster_names:
                scored_gpt_cluster_names[clus] = list(gpt_cluster_names[clus])

        used_cluster_names = set([])
        for cluster_id in scored_gpt_cluster_names:
            cluster_name_options = list(scored_gpt_cluster_names[cluster_id])
            for cluster_name in cluster_name_options:
                if cluster_name in used_cluster_names:
                    continue
                else:
                    cluster_names[cluster_id] = cluster_name
                    used_cluster_names.add(cluster_name)
                    break

        missing_name_keys = [key for key in cluster_df["cluster_id"].to_list() if key not in cluster_names]
        failure_count = 0
        for key in missing_name_keys:
            entities = list(set(cluster_entity_dict[key]) - used_cluster_names)
            picked_entitiy = None
            if len(entities) > 0:
                picked_entitiy = random.choice(entities)

            if picked_entitiy is None:
                failure_count += 1
                picked_entitiy = f"unnamed cohort {failure_count}"

            cluster_names[key] = picked_entitiy
            used_cluster_names.add(picked_entitiy)

        return cluster_names

    def compute_cluster_name_relevance(self, cluster_name_dict, cluster_entity_dict):
        t_init = time.time()
        cluster_options = []
        keyword_options = []
        for clus_id in cluster_entity_dict:
            cluster_options.extend(list(cluster_name_dict.get(clus_id, [])))
            keyword_options.extend(list(cluster_entity_dict.get(clus_id, [])))

        cluster_options = list(set(cluster_options))
        keyword_options = list(set(keyword_options))
        cluster_entitities = list([k for k in cluster_entity_dict])

        cluster_entitities_idx_map = {k: i for i, k in enumerate(cluster_entitities)}
        cluster_options_idx_map = {k: i for i, k in enumerate(cluster_options)}
        keyword_options_idx_map = {k: i for i, k in enumerate(keyword_options)}
        cluster_entitities_keywords_ids = {
            clus: [keyword_options_idx_map[kw] for kw in cluster_entity_dict[clus]] for clus in cluster_entity_dict
        }

        all_keywords_ordered = sorted(set([*cluster_options, *keyword_options]))
        embeddings = bert_api_async(all_keywords_ordered, ST_BERT_CONFIG, model="MiniLML6V2", chunk_size=500)
        embeddings_dict = dict(zip(all_keywords_ordered, embeddings))

        t_fetch_embeddings = time.time()
        print("Time taken for fetching embeddings : ", t_fetch_embeddings - t_init)

        name_sim_matrix = np.full(
            (len(cluster_options), len(keyword_options)),
            fill_value=0,
            dtype=np.float16,
        )

        chunk_size = 10000
        clus_chunks = list(chunks(cluster_options, chunk_size=chunk_size))
        kws_chunks = list(chunks(keyword_options, chunk_size=chunk_size))

        for row_index in tqdm(range(len(clus_chunks)), desc="Row completion", leave=True):
            for col_index in tqdm(range(len(kws_chunks)), desc="Col completion", leave=False):
                row_chunk = clus_chunks[row_index]
                col_chunk = kws_chunks[col_index]
                row_idx_start = row_index * chunk_size
                col_idx_start = col_index * chunk_size

                mini_matrix = cosine_similarity(
                    [embeddings_dict[k] for k in row_chunk], [embeddings_dict[k] for k in col_chunk]
                ).astype(np.float16)

                name_sim_matrix[
                    row_idx_start : row_idx_start + len(row_chunk),
                    col_idx_start : col_idx_start + len(col_chunk),
                ] = mini_matrix

        @np.vectorize
        def agg_text_score_func(x, y):
            return np.mean(name_sim_matrix[np.ix_([x], cluster_entitities_keywords_ids[cluster_entitities[y]])])

        cohort_names_cohort_sim_matrix = np.fromfunction(
            function=agg_text_score_func, shape=(len(cluster_options), len(cluster_entitities)), dtype=int
        ).astype(np.float16)

        scored_cluster_name_dict = {}
        for clus in cluster_name_dict:
            clus_idx = cluster_entitities_idx_map[clus]
            for clus_name in cluster_name_dict[clus]:
                clus_name_idx = cluster_options_idx_map[clus_name]
                score = cohort_names_cohort_sim_matrix[clus_name_idx, clus_idx]

                if clus not in scored_cluster_name_dict:
                    scored_cluster_name_dict[clus] = {}

                scored_cluster_name_dict[clus][clus_name] = score
        return scored_cluster_name_dict

    def group_cluster(self, targetings):
        """Group the targetings level data to cluster level data using the cohort_id

        Args:
            targetings (pd.DataFrame): Targetings data frame with cohort id

        Returns:
            pd.DataFrame: Cohorts dataframe
        """
        t_init = time.time()
        targetings_data_dict = targetings.set_index("keyword")[["relevance_score"]].to_dict()
        cluster_df = (
            targetings.groupby("cluster_id")
            .agg({"keyword": list, "idx": list, "cohort_type": "first"})
            .reset_index()
            .rename(columns={"keyword": "keywords", "idx": "keywords_idx"})
        )
        cluster_df = cluster_df[cluster_df["cohort_type"] == "regular"]

        cluster_df["cohort_size"] = cluster_df["keywords"].map(len)

        cluster_df["relevance_score"] = cluster_df["keywords"].map(
            lambda x: compute_mean([targetings_data_dict["relevance_score"][t] for t in x], skipna=True)
        )
        cluster_df = cluster_df.sort_values(["relevance_score"], ascending=False)
        cluster_df["cluster_id"] = [str(i + 1) for i in range(len(cluster_df))]

        similar_cohort_dict = self.fetch_similar_cohorts(cluster_df[["cluster_id", "keywords_idx"]])
        cluster_df["similar_clusters"] = cluster_df["cluster_id"].map(lambda x: similar_cohort_dict.get(x, {}))

        req_cols = [
            "cluster_id",
            "keywords",
            "keywords_idx",
            "cohort_size",
            "relevance_score",
            "similar_clusters",
        ]
        cluster_df = cluster_df.sort_values(["relevance_score"], ascending=False)
        cluster_df = cluster_df[req_cols]
        print("Time taken to group clusters : ", time.time() - t_init)
        return cluster_df

    def fetch_similar_cohorts(self, cluster_df: pd.DataFrame):
        cluster_df = cluster_df.copy(deep=True)[["cluster_id", "keywords_idx"]]
        cohort_sim_df = cluster_df.merge(cluster_df, how="cross", suffixes=("_a", "_b"))

        cohort_sim_df["cluster_similarity"] = cohort_sim_df[["keywords_idx_a", "keywords_idx_b"]].apply(
            lambda x: group_sim_from_matrix(self.sim_matrix[np.ix_(x[0], x[1])]), axis=1, result_type="reduce"
        )

        sim_cohorts_df = (
            cohort_sim_df.groupby("cluster_id_a").agg({"cluster_id_b": list, "cluster_similarity": list}).reset_index()
        )
        sim_cohorts_df["similar_clusters"] = sim_cohorts_df[["cluster_id_b", "cluster_similarity"]].apply(
            lambda x: dict(zip(x[0], x[1])), axis=1, result_type="reduce"
        )
        sim_cohorts_df["similar_clusters"] = sim_cohorts_df[["cluster_id_a", "similar_clusters"]].apply(
            lambda x: {
                k: x[1][k]
                for k in sorted(x[1], key=x[1].get, reverse=True)
                if (k != x[0]) and (x[1][k] >= ClusterKeywords.similarity_cutoff)
            },
            axis=1,
            result_type="reduce",
        )

        sim_cohorts_dict = sim_cohorts_df.set_index("cluster_id_a")["similar_clusters"].to_dict()
        return sim_cohorts_dict

    def clean_entities(self):
        self.raw_keywords_df["keyword_name"] = self.raw_keywords_df["keyword"]
        lemmatized_keywords = self.lemmatizer.lemmatize(kw_list=self.raw_keywords_df["keyword_name"].to_list())
        self.raw_keywords_df["keyword_name"] = lemmatized_keywords
        self.raw_keywords_df["keyword_name"] = self.raw_keywords_df["keyword_name"].map(clean_keywords)

        self.keywords_df = self.raw_keywords_df.copy(deep=True)
        self.keywords_df = self.keywords_df[self.keywords_df["keyword_name"].str.len() > 0]
        self.keywords_df = self.keywords_df.drop_duplicates(subset=["keyword_name"])

    def create_sub_phrases(self):
        self.keywords_df["split_words"] = self.keywords_df["keyword_name"].map(lambda x: x.split(" "))
        self.keywords_df["word_count"] = self.keywords_df["split_words"].map(len)
        self.keywords_df["sub_phrases"] = self.keywords_df["split_words"].map(
            lambda x: [
                [" ".join(x[start : start + l]) for start in range(0, len(x) + 1 - l)] for l in range(1, len(x) + 1)
            ]
        )
        self.keywords_df["sub_phrases"] = self.keywords_df["sub_phrases"].map(lambda x: list(chain.from_iterable(x)))

    def create_keyword_ids(self):
        kw_list = list(
            set(
                [
                    *self.keywords_df["keyword_name"].to_list(),
                    *self.keywords_df["sub_phrases"].explode().drop_duplicates().to_list(),
                ]
            )
        )

        self.keyword_id_name_link = dict(zip([str(i) for i in range(len(kw_list))], kw_list))
        self.keyword_name_id_link = {k: i for i, k in self.keyword_id_name_link.items()}

        self.keywords_df["id"] = self.keywords_df["keyword_name"].map(lambda x: self.keyword_name_id_link.get(x))
        self.keywords_df["sub_phrases_id"] = self.keywords_df["sub_phrases"].map(
            lambda x: [self.keyword_name_id_link.get(p) for p in x]
        )

        self.keyword_sub_phrases_link = self.keywords_df.set_index("keyword_name")["sub_phrases"].to_dict()
        self.to_cluster_kw = self.keywords_df["keyword_name"].to_list()

    def compute_phrases_similarity(self):
        print("Fetching phrases similarity")
        t_init = time.time()

        all_keywords_ordered = sorted(list(self.keyword_name_id_link.keys()))
        kws_idx_dict = dict(zip(all_keywords_ordered, range(len(all_keywords_ordered))))

        embeddings = bert_api_async(all_keywords_ordered, ST_BERT_CONFIG, model="MiniLML6V2", chunk_size=500)
        embeddings_dict = dict(zip(all_keywords_ordered, embeddings))

        t_fetch_embeddings = time.time()
        print("Time taken for fetching embeddings : ", t_fetch_embeddings - t_init)

        phrases_sim_matrix = np.full(
            (len(all_keywords_ordered), len(all_keywords_ordered)),
            fill_value=0,
            dtype=np.float16,
        )

        chunk_size = 10000
        kws_chunks = list(chunks(all_keywords_ordered, chunk_size=chunk_size))

        for row_index in tqdm(range(len(kws_chunks)), desc="Row completion", leave=True):
            for col_index in tqdm(range(len(kws_chunks)), desc="Col completion", leave=False):
                scoring_kws_chunk = kws_chunks[row_index]
                adg_kws_chunk = kws_chunks[col_index]
                row_idx_start = row_index * chunk_size
                col_idx_start = col_index * chunk_size

                mini_matrix = cosine_similarity(
                    [embeddings_dict[k] for k in scoring_kws_chunk], [embeddings_dict[k] for k in adg_kws_chunk]
                ).astype(np.float16)

                phrases_sim_matrix[
                    row_idx_start : row_idx_start + len(scoring_kws_chunk),
                    col_idx_start : col_idx_start + len(adg_kws_chunk),
                ] = mini_matrix

        t_sim_computation = time.time()
        print("Time taken for semantic computation : ", t_sim_computation - t_fetch_embeddings)

        kw_list = self.to_cluster_kw
        kw_count = len(kw_list)
        new_idx_old_idx_map = {i: kws_idx_dict[kw] for i, kw in enumerate(kw_list)}
        sub_phrases_old_idx_dict = {
            i: [kws_idx_dict[sp] for sp in self.keyword_sub_phrases_link[kw]] for i, kw in enumerate(kw_list)
        }

        @np.vectorize
        def kw_score_func(x, y):
            return phrases_sim_matrix[new_idx_old_idx_map[x], new_idx_old_idx_map[y]]

        @np.vectorize
        def agg_phrases_score_func(x, y):
            return group_sim_from_matrix(
                phrases_sim_matrix[np.ix_(sub_phrases_old_idx_dict[x], sub_phrases_old_idx_dict[y])]
            )

        keyword_sim_matrix = np.fromfunction(kw_score_func, (kw_count, kw_count)).astype(np.float16)
        keyword_phrases_sim_matrix = np.fromfunction(agg_phrases_score_func, (kw_count, kw_count)).astype(np.float16)

        del phrases_sim_matrix
        sim_matrix = (
            ((1 - self.subphrases_sim_weight) * keyword_sim_matrix)
            + (self.subphrases_sim_weight * keyword_phrases_sim_matrix)
        ).astype(np.float16)
        del keyword_sim_matrix
        del keyword_phrases_sim_matrix

        self.matrix_kw_order = kw_list
        self.sim_matrix = sim_matrix
        t_complex_sim_computation = time.time()
        print("Time taken for complex semantic computation : ", t_complex_sim_computation - t_sim_computation)


class Comprehensive_Clustering:
    """Comprehensive Clustering algorithm
    Uses multiple iterations of Agglomerative cluster to create cohorts considering ideal cohort size.
    Starts clustering with lower cutoff upto max cutoff. Yields good cohorts during initial rounds of clustering.
    """

    def __init__(
        self,
        tar_sim_matrix,
        tar_id_order,
        recm_clus_size=10,
        min_clus_size=3,
        distance_step=0.03,
        clustering_distance_cutoff=0.25,
        name_dict={},
    ):
        self.tar_sim_matrix = tar_sim_matrix
        self.tar_dist_matrix = 1 - tar_sim_matrix
        self.tar_id_order = tar_id_order
        self.name_dict = name_dict
        self.tar_id_idx_dict = {id: i for i, id in enumerate(self.tar_id_order)}
        self.targetings = pd.DataFrame({"id": self.tar_id_order})
        self.recm_clus_size = recm_clus_size
        self.min_clus_size = min_clus_size
        self.distance_step = distance_step
        self.clustering_distance_cutoff = clustering_distance_cutoff
        self.min_sim_cutoff = 1 - (clustering_distance_cutoff * 1.0)
        self.adaptive_clustering()

    def change_analysis(self, v1, v2):
        """Debugger method to track changes over iterations
        Useful while tuning the params
        Args:
            v1 (pd.DataFrame): Cohort v1
            v2 (pd.DataFrame): Cohort v2

        Returns:
            pd.DataFrame: Cohort comparison dataframe
        """
        cohorts_v1 = v1.groupby("cohort_id").agg({"id": list}).reset_index().rename(columns={"id": "c1_tar"})
        cohorts_v2 = v2.groupby("cohort_id").agg({"id": list}).reset_index().rename(columns={"id": "c2_tar"})

        cohorts_v1["c1_tar_name"] = cohorts_v1["c1_tar"].map(lambda x: [self.name_dict[c] for c in x])
        cohorts_v2["c2_tar_name"] = cohorts_v2["c2_tar"].map(lambda x: [self.name_dict[c] for c in x])

        cohorts = cohorts_v1.merge(cohorts_v2, on="cohort_id", how="outer")
        cohorts["c1_tar_name"] = cohorts["c1_tar_name"].map(lambda x: x if isinstance(x, list) else [])
        cohorts["c2_tar_name"] = cohorts["c2_tar_name"].map(lambda x: x if isinstance(x, list) else [])

        # get added targetings
        cohorts["addition"] = cohorts[["c1_tar_name", "c2_tar_name"]].apply(
            lambda x: list(set(x[1]) - set(x[0])), axis=1, result_type="reduce"
        )
        # get removed targetings
        cohorts["removal"] = cohorts[["c1_tar_name", "c2_tar_name"]].apply(
            lambda x: list(set(x[0]) - set(x[1])), axis=1, result_type="reduce"
        )
        # get common targetings
        cohorts["common"] = cohorts[["c1_tar_name", "c2_tar_name"]].apply(
            lambda x: list(set(x[0]).union(set(x[1]))), axis=1, result_type="reduce"
        )
        return cohorts

    def adaptive_clustering(self):
        """Algorithm
        Slab clustering --> Slab clustering algorithm, uses iterative Agglomerative clustering
            1:Starts with min cutoff
            2:Clusters the targetings
            3:Puts aside the targetings clustered
            (unclustered targetings are the ones in the cohort with size less than recommended cohorts size)
            4:Clusters unclustered targetings with the next cutoff
            5:Repeats point 3 and 4 until max cutoff is reached
            6:Combine the results

        Reclustering --> Shuffle the targetings among cohorts if any targeting in a cohort is more closer to a different cohort.
        This might happen due to clustering at mutliple cutoffs

        Split the clusters --> If the clusters are too big then split them to bring them closer to recommended size.
        This is done by using agglomerative clustering on the big cluster to split them into n components
        n if computed as cohort length/recommended cohort size

        Combine clusters --> Combine clusters that are too small (below min cluster size)


        For this implementation the operations order are:
        1: Perform slab clustering --> gets initial clusters
        2: Split the clusters --> splits big clusters
        3: Recluster the targetings --> shuffle to targetings
        4: Split the clusters --> split the cluster in case cohort got too big after reclustering
        5: Combine the clusters --> combine the cluster to avoid less solo targetings
        """
        t_init = time.time()
        clustering_out_dict = self.slab_clustering()

        self.targetings["cohort_id"] = self.targetings["id"].map(lambda x: clustering_out_dict["cohort_id"][x])
        self.targetings["clus_cutoff"] = self.targetings["id"].map(lambda x: clustering_out_dict["clus_cutoff"][x])

        # no cluster splitting as for us there is no point splitting similar keywords just to form bigger adgroups

        # reclustering
        self.reclustering()

        # combine small cohorts
        self.combine_clusters()

        # format clusters
        self.cluster_formatting()

        t_end = time.time()
        print(f"Time taken for comprehensive clustering : {t_end - t_init}")

    def slab_clustering(self):
        """Slab clustering algorithm, uses iterative Agglomerative clustering
        1:Starts with min cutoff
        2:Clusters the targetings
        3:Puts aside the targetings clustered
        (unclustered targetings are the ones in the cohort with size less than recommended cohorts size)
        4:Clusters unclustered targetings with the next cutoff
        5:Repeats point 3 and 4 until max cutoff is reached
        6:Combine the results
        Returns:
            dict: Dict with keys as targetings id and value as corresponding cohort id and the cutoff at which it was clustered
        """
        print("Initiating slab clustering algo..")
        t_init = time.time()

        cluster_class_df = pd.DataFrame({"id": self.tar_id_order})
        cluster_class_df["idx"] = cluster_class_df["id"].map(lambda x: self.tar_id_idx_dict[x])
        cluster_class_df["cohort_id"] = None
        cluster_class_df["clus_size"] = 0
        cluster_class_df["clus_cutoff"] = 0
        # start with min cutoff
        start_cutoff = max(0.05, self.distance_step)

        # list of all cutoffs to use for agglomerative clustering
        all_iters_cutoffs = list(
            np.arange(
                start=start_cutoff,
                stop=self.clustering_distance_cutoff,
                step=self.distance_step,
            )
        )
        if len(all_iters_cutoffs) == 0:
            all_iters_cutoffs = [self.clustering_distance_cutoff]

        # append all iters with max cutoff to end with
        if max(all_iters_cutoffs) < self.clustering_distance_cutoff:
            all_iters_cutoffs.append(self.clustering_distance_cutoff)

        # iterative clustering
        min_clus_id = 1
        for iter, cutoff in enumerate(all_iters_cutoffs):
            # pick all targetings with clus size less than recommended cohorts size
            tar_ids_to_cluster = cluster_class_df[cluster_class_df["clus_size"] < self.recm_clus_size]["id"]
            # create matrix for the targetings to be clustered
            tar_dist_matrix_to_cluster = self.tar_dist_matrix[
                np.ix_(
                    [self.tar_id_idx_dict[c] for c in tar_ids_to_cluster],
                    [self.tar_id_idx_dict[c] for c in tar_ids_to_cluster],
                )
            ]

            # cluster the targetings
            print(f"Executing iter {iter} with cutoff {cutoff} for {len(tar_ids_to_cluster)} targetings..")
            cluster_class_dict = self.agglomerative_clustering(
                dist_matrix=tar_dist_matrix_to_cluster,
                id_order=tar_ids_to_cluster,
                distance_threshold=cutoff,
                min_clus_id=min_clus_id,
            )

            # allocate cohort id
            cluster_class_df["cohort_id"] = cluster_class_df[["cohort_id", "id"]].apply(
                lambda x: x[0] if x[1] not in cluster_class_dict else cluster_class_dict[x[1]],
                axis=1,
                result_type="reduce",
            )
            cluster_class_df["clus_cutoff"] = cluster_class_df[["clus_cutoff", "id"]].apply(
                lambda x: x[0] if x[1] not in cluster_class_dict else cutoff,
                axis=1,
                result_type="reduce",
            )

            min_clus_id = cluster_class_df["cohort_id"].max()
            # get cluster size
            cluster_class_df["clus_size"] = cluster_class_df.groupby("cohort_id").transform("count")["id"]

        t_end = time.time()
        print(f"Time taken for slab clustering algo execution : {t_end - t_init}")

        out_dict = cluster_class_df.set_index("id")[["clus_cutoff", "cohort_id"]].to_dict()
        return out_dict

    def reclustering(self):
        """Reclusters the targetings
        Shuffle the targetings among cohorts if any targeting in a cohort is more closer to a different cohort.
        This might happen due to clustering at mutliple cutoffs
        """
        print("Initiating reclustering algo..")
        t_init = time.time()
        all_cohorts_dict = self.targetings.groupby("cohort_id").agg({"id": list})["id"].to_dict()
        cohort_ids = [c for c in all_cohorts_dict if len(all_cohorts_dict[c]) > 1]
        if len(cohort_ids) == 0:
            return None
        cohort_id_tar_list = [all_cohorts_dict[c] for c in cohort_ids]
        # for each targetings, tag all targetings in that cohort
        self.targetings["cohort_id_list"] = self.targetings["id"].map(lambda x: cohort_id_tar_list)
        # compute similarity between the targetings and all the targetings of the other cohorts
        self.targetings["cohort_tar_sims"] = self.targetings["id"].map(
            lambda x: [
                self.tar_sim_matrix[
                    np.ix_(
                        [self.tar_id_idx_dict[x]],
                        [self.tar_id_idx_dict[c] for c in cohort_id_tar],
                    )
                ][0]
                for cohort_id_tar in cohort_id_tar_list
            ]
        )
        top_n_perc = 1.0
        # Create a cohort similarity score with the targeting
        # For each targeting, pick top n perc closest targetings of the cohort and take the mean
        self.targetings["cohort_sim"] = self.targetings["cohort_tar_sims"].map(
            lambda x: [c[np.argsort(c)][-int(np.ceil(c.shape[0] * top_n_perc)) :].mean() for c in x]
        )
        self.targetings["closest_cohort_id"] = self.targetings["cohort_sim"].map(lambda x: cohort_ids[np.argmax(x)])
        self.targetings["closest_cohort_sim"] = self.targetings["cohort_sim"].map(lambda x: np.max(x))
        # check if closest cohort is different that targeting's current cohort
        self.targetings["closest_cohort_dif"] = self.targetings["closest_cohort_id"] != self.targetings["cohort_id"]

        # if targetings current cohort is different than the closest one and similairty is more than the cutoff then shuffle
        targetings_with_cohort_change = self.targetings[
            (self.targetings["closest_cohort_dif"] == True)
            & (self.targetings["closest_cohort_sim"] > self.min_sim_cutoff)
        ]
        targetings_with_cohort_change_dict = targetings_with_cohort_change.set_index("id")[
            "closest_cohort_id"
        ].to_dict()

        print(f"Targetings with cohort change : {len(targetings_with_cohort_change_dict)}")
        self.targetings["cohort_id"] = self.targetings[["cohort_id", "id"]].apply(
            lambda x: targetings_with_cohort_change_dict[x[1]] if x[1] in targetings_with_cohort_change_dict else x[0],
            axis=1,
            result_type="reduce",
        )

        self.targetings = self.targetings[["id", "cohort_id", "clus_cutoff"]]
        t_end = time.time()
        print(f"Time taken for reclustering algo execution : {t_end - t_init}")

    def cluster_splitting(self):
        """Split the clusters
        If the clusters are too big then split them to bring them closer to recommended size.
        This is done by using agglomerative clustering on the big cluster to split them into n components
        n if computed as cohort length/recommended cohort size
        """
        print("Initiating cluster splitting algo..")
        t_init = time.time()
        min_clus_id = self.targetings["cohort_id"].max() + 1
        all_cohorts_dict = self.targetings.groupby("cohort_id").agg({"id": list})["id"].to_dict()
        # consider the cohort that are atleast 1.5x of the recommended cohort size
        cohort_ids = [c for c in all_cohorts_dict if len(all_cohorts_dict[c]) > int(self.recm_clus_size * 1.5)]
        if len(cohort_ids) == 0:
            return None

        cohort_df = pd.DataFrame({"cohort_id": cohort_ids})
        cohort_df["targetings_ids"] = cohort_df["cohort_id"].map(lambda c: all_cohorts_dict[c])
        # define ideal subcluster count
        cohort_df["num_sub_cohorts"] = cohort_df["targetings_ids"].map(
            lambda x: int(np.round(len(x) / self.recm_clus_size))
        )
        cohort_df = cohort_df[cohort_df["num_sub_cohorts"] > 1]
        if len(cohort_df) == 0:
            return None

        cohort_df["dist_matrix"] = cohort_df["targetings_ids"].map(
            lambda x: self.tar_dist_matrix[
                np.ix_(
                    [self.tar_id_idx_dict[t] for t in x],
                    [self.tar_id_idx_dict[t] for t in x],
                )
            ]
        )
        # cluster the targeting using agglomerative cluster with ideal n_clusters as param
        cohort_df["cluster_class"] = cohort_df[["dist_matrix", "targetings_ids", "num_sub_cohorts"]].apply(
            lambda x: self.agglomerative_clustering(dist_matrix=x[0], id_order=x[1], n_clusters=x[2], min_clus_id=1),
            axis=1,
            result_type="reduce",
        )
        if len(self.name_dict) > 0:
            cohort_df["cluster_class_with_names"] = cohort_df["cluster_class"].map(
                lambda x: {self.name_dict[c]: x[c] for c in x}
            )
        # allocated new cluster ids
        min_clus_id = min_clus_id + 2
        targetings_with_cohort_change_dict = {}
        for cohort_cluster_classes in cohort_df["cluster_class"].to_list():
            for tar in cohort_cluster_classes:
                targetings_with_cohort_change_dict[tar] = cohort_cluster_classes[tar] + min_clus_id
            min_clus_id = max(max(targetings_with_cohort_change_dict.values()), min_clus_id) + 1

        print(f"Targetings with cohort split : {len(targetings_with_cohort_change_dict)}")
        self.targetings["cohort_id"] = self.targetings[["cohort_id", "id"]].apply(
            lambda x: targetings_with_cohort_change_dict[x[1]] if x[1] in targetings_with_cohort_change_dict else x[0],
            axis=1,
            result_type="reduce",
        )
        self.targetings = self.targetings[["id", "cohort_id", "clus_cutoff"]]
        t_end = time.time()
        print(f"Time taken for clustering splitting algo execution : {t_end - t_init}")

    def combine_clusters(self):
        """Combine clusters
        Combine clusters that are too small (below min cluster size)"""
        print("Initiating cluster combination algo..")
        t_init = time.time()
        all_cohorts_dict = self.targetings.groupby("cohort_id").agg({"id": list})["id"].to_dict()
        cohort_ids_to_keep = [c for c in all_cohorts_dict if len(all_cohorts_dict[c]) >= self.min_clus_size]
        cohort_ids_to_combine = [c for c in all_cohorts_dict if len(all_cohorts_dict[c]) < self.min_clus_size]

        if (len(cohort_ids_to_combine) == 0) | (len(cohort_ids_to_keep) == 0):
            return None

        # consider cohorts that are smaller than min size
        cohort_to_keep_ids = [all_cohorts_dict[c] for c in cohort_ids_to_keep]

        cohort_to_comb_df = pd.DataFrame({"cohort_id": cohort_ids_to_combine})
        cohort_to_comb_df["tar_id_list"] = cohort_to_comb_df["cohort_id"].map(lambda x: all_cohorts_dict[x])
        # compute inter cohort sim matrix
        cohort_to_comb_df["inter_cohort_sim_matrix"] = cohort_to_comb_df["tar_id_list"].map(
            lambda x: [
                self.tar_sim_matrix[
                    np.ix_(
                        [self.tar_id_idx_dict[c] for c in x],
                        [self.tar_id_idx_dict[c] for c in cohort_tar_ids],
                    )
                ]
                for cohort_tar_ids in cohort_to_keep_ids
            ]
        )

        # get an average sim number from the matrix for each cohort
        cohort_to_comb_df["inter_cohort_sims"] = cohort_to_comb_df["inter_cohort_sim_matrix"].map(
            lambda x: [group_sim_from_matrix_percentile(matrix=cohort_sim, top_n_perc=1.0) for cohort_sim in x]
        )

        # find closest cohort and sim
        cohort_to_comb_df["closest_cohort_id"] = cohort_to_comb_df["inter_cohort_sims"].map(
            lambda x: cohort_ids_to_keep[np.argmax(x)]
        )
        cohort_to_comb_df["closest_cohort_sim"] = cohort_to_comb_df["inter_cohort_sims"].map(lambda x: np.max(x))

        # merge cohorts if similarity is more than the min cutoff
        close_cohorts = cohort_to_comb_df[cohort_to_comb_df["closest_cohort_sim"] > self.min_sim_cutoff]
        targetings_with_cohort_change_dict = (
            close_cohorts[["tar_id_list", "cohort_id", "closest_cohort_id"]]
            .explode("tar_id_list")
            .set_index("tar_id_list")["closest_cohort_id"]
            .to_dict()
        )

        if len(self.name_dict) > 0:
            close_cohorts["tar_names"] = close_cohorts["tar_id_list"].map(lambda x: [self.name_dict[c] for c in x])
            close_cohorts["closest_cohort_tar_names"] = close_cohorts["closest_cohort_id"].map(
                lambda x: [self.name_dict[c] for c in all_cohorts_dict[x]]
            )
        print(f"Targetings with cohort change : {len(targetings_with_cohort_change_dict)}")

        self.targetings["cohort_id"] = self.targetings[["cohort_id", "id"]].apply(
            lambda x: targetings_with_cohort_change_dict[x[1]] if x[1] in targetings_with_cohort_change_dict else x[0],
            axis=1,
            result_type="reduce",
        )
        self.targetings = self.targetings[["id", "cohort_id", "clus_cutoff"]]
        t_end = time.time()
        print(f"Time taken for cluster combination algo execution : {t_end - t_init}")

    def agglomerative_clustering(
        self,
        dist_matrix,
        id_order,
        distance_threshold=None,
        n_clusters=None,
        min_clus_id=1,
    ):
        """Implementation of standard agglomerative clustering

        Args:
            dist_matrix (np.ndarray): Distance matrix
            id_order (list): ID order corresponding to the matrix
            distance_threshold (float, optional): Distance threshold for clustering. Defaults to None.
            n_clusters (int, optional): Num clusters for clustering. Not needed if distance threshold is passed. Defaults to None.
            min_clus_id (int, optional): Minimum cluster id, allocated ids on top of this number. Defaults to 1.

        Raises:
            ClusteringError: Raises an exception if both distance_threshold and n_clusters are None or if both are non None
            ClusteringError: Raises an exception if any issue is encountered during clustering

        Returns:
            dict: Dict with targeting id as key and cluster id as value
        """
        t_init = time.time()
        if (distance_threshold is None) & (n_clusters is None):
            raise ClusteringError(message="Either of distance_threshold or n_clusters must be non-None")
        if (distance_threshold is not None) & (n_clusters is not None):
            raise ClusteringError(message="Either of distance_threshold or n_clusters must be None")

        if dist_matrix.shape[0] <= 1:
            data_vec_class = [min_clus_id + 1 for _ in id_order]
            return dict(zip(id_order, data_vec_class))
        clustering_type, clustering_val = None, None
        if distance_threshold is not None:
            clustering_type = "cutoff"
            clustering_val = distance_threshold
        if n_clusters is not None:
            clustering_type = "clusters count"
            clustering_val = n_clusters
        try:
            aggcl = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric="precomputed",
                linkage="average",
                distance_threshold=distance_threshold,
            )
            data_vec_class = aggcl.fit_predict(dist_matrix) + min_clus_id + 1
            cluster_class_dict = dict(zip(id_order, data_vec_class))
        except Exception as e:
            raise ClusteringError(message=str(e))
        t_end = time.time()
        print(
            f"Time taken for cutoff based agglomerative clustering with {len(id_order)} elements and {clustering_val} {clustering_type}  : {t_end - t_init}"
        )
        return cluster_class_dict

    def cluster_formatting(self):
        """Creates cluster dataframe from targetings dataframe"""
        t_init = time.time()
        self.targetings["clus_size"] = self.targetings.groupby("cohort_id").transform("count")["id"]

        self.targetings["cohort_type"] = self.targetings["clus_size"].map(
            lambda x: "regular" if (x >= self.min_clus_size) else "others"
        )
        self.targetings["cohort_id"] = self.targetings[["cohort_id", "cohort_type"]].apply(
            lambda x: x[0] if (x[1] == "regular") else (self.targetings["cohort_id"].max() + 1),
            axis=1,
            result_type="reduce",
        )

        cluster_df = (
            self.targetings.groupby("cohort_id")
            .agg({"id": list, "cohort_type": "first", "clus_cutoff": "max"})
            .reset_index()
            .rename(columns={"id": "targetings_ids"})
        )
        cluster_df["cohort_size"] = cluster_df["targetings_ids"].map(len)

        cluster_df = cluster_df.sort_values(["cohort_size"], ascending=False)
        cluster_df["cohort_id"] = [str(i + 1) for i in range(len(cluster_df))]
        t_end = time.time()
        print(f"Time_taken for cluster formatting : {t_end - t_init}")
        self.cluster_df = cluster_df
