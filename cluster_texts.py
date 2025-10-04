import os
import string
import time

import nltk
import numpy as np
import pandas as pd
import unidecode
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph
from sklearn.neighbors._graph import _query_include_self

nltk_data_dir = os.getenv("nltk_data_dir", "/tmp/nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)

# Add the custom directory to NLTK's data path
if nltk_data_dir not in nltk.data.path:
    nltk.data.path.append(nltk_data_dir)

if not os.path.exists(os.path.join(nltk_data_dir, "corpora", "stopwords")):
    nltk.download("stopwords", download_dir=nltk_data_dir)


def text_process_uni_bi(text: str) -> str:
    """Preprocess/clean keywords

    Args:
        text (str): Raw keyword

    Returns:
        str: Clean keywords
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
    nopunc_bi = [" ".join(c) for c in nltk.bigrams(nopunc)]
    nopunc.extend(nopunc_bi)
    return nopunc


def NNGraph_fetch_neighbours(
    data: pd.DataFrame, x_transformed, dist_threshold=0.7, max_neighbors=100
):
    """
    Perform agglomerative clustering on the given data.

    Args:
        data (DataFrame): Input data.
        x_transformed (): Transformed input data array-like.

    Returns:
        DataFrame: DataFrame with cluster IDs and sizes.
    """
    t_init = time.time()

    df = data.copy().reset_index(drop=True)
    df_text_dict = df["text"].to_dict()

    # TODO adjust this after output review
    max_neighbors = min(max_neighbors, int(np.sqrt(data.shape[0] - 1)))

    # do nothing is upto 2 entries
    if len(data) <= 2:
        req_cols_default_dict = {
            "neighbors": [],
            "neighbor_names": [],
            "neighbor_dist": [],
            "neighbor_count": 0,
        }
        for col in df:
            df[col] = df["text"].map(lambda _: req_cols_default_dict[col])
        return df

    X = NearestNeighbors(
        n_neighbors=max_neighbors,
        radius=dist_threshold,
        metric="euclidean",
        metric_params=None,
        n_jobs=None,
    ).fit(x_transformed)

    t_nn = time.time()
    print("Total time for graph : ", t_nn - t_init)

    query_distance = _query_include_self(X._fit_X, include_self=False, mode="distance")
    query_connectivity = _query_include_self(
        X._fit_X, include_self=False, mode="connectivity"
    )
    # find distance between connections
    neigh_knn_graph_dist = X.kneighbors_graph(
        X=query_distance,
        n_neighbors=max_neighbors,
        mode="distance",
    )

    t_dist = time.time()
    print("Total time for dist : ", t_dist - t_nn)

    # find connections
    neigh_knn_graph_conn = X.kneighbors_graph(
        X=query_connectivity, n_neighbors=max_neighbors, mode="connectivity"
    )
    t_con = time.time()
    print("Total time for con : ", t_con - t_dist)

    # faster custom implementation, implement later after testing, modify X if needed
    # computes conn and dist in single run

    # n_neighbors=max_neighbors
    # A_data, A_ind = X.kneighbors(n_neighbors, return_distance=True)
    # A_data = np.ravel(A_data)
    # n_queries = A_ind.shape[0]
    # A_data_ones=np.ones(n_queries * n_neighbors)
    # n_samples_fit = X.n_samples_fit_
    # n_nonzero = n_queries * n_neighbors
    # A_indptr = np.arange(0, n_nonzero + 1, n_neighbors)

    # neigh_knn_graph_dist = csr_matrix(
    #         (A_data, A_ind.ravel(), A_indptr), shape=(n_queries, n_samples_fit)
    #     )
    # neigh_knn_graph_conn = csr_matrix(
    #         (A_data_ones, A_ind.ravel(), A_indptr), shape=(n_queries, n_samples_fit)
    #     )

    # pull connected elements
    non_zero_elements = neigh_knn_graph_conn.nonzero()
    # put a limit of similarity radius
    dist_condition = np.abs(neigh_knn_graph_dist.data) <= dist_threshold

    # find all connections that follow distance cutoff
    non_zero_data = np.squeeze(np.asarray(neigh_knn_graph_dist[non_zero_elements]))[
        dist_condition
    ]
    non_zero_rows = non_zero_elements[0][dist_condition]
    non_zero_cols = non_zero_elements[1][dist_condition]

    # parse neighbors to df
    # some cols are unnecessary, remove after few rounds of review
    val_df = pd.DataFrame(
        {
            "entity_id": non_zero_rows,
            "neighbor_id": non_zero_cols,
            "neighbor_dist": non_zero_data,
        }
    )
    val_df["neighbor_name"] = val_df["neighbor_id"].map(lambda x: df_text_dict.get(x))
    val_df["text"] = val_df["entity_id"].map(lambda x: df_text_dict.get(x))
    val_df = (
        val_df.groupby("text")
        .agg(
            neighbors=("neighbor_id", lambda x: list(x)),
            neighbor_names=("neighbor_name", lambda x: list(x)),
            neighbor_dist=("neighbor_dist", lambda x: list(x)),
        )
        .reset_index()
    )

    df = df.merge(val_df, on="text", how="left")

    df["neighbors"] = df["neighbors"].map(lambda x: x if isinstance(x, list) else [])
    df["neighbor_names"] = df["neighbor_names"].map(
        lambda x: x if isinstance(x, list) else []
    )
    df["neighbor_dist"] = df["neighbor_dist"].map(
        lambda x: x if isinstance(x, list) else []
    )
    df["neighbor_count"] = df["neighbors"].map(lambda x: len(x))

    t_end = time.time()
    print("Total time for mapping : ", t_end - t_con)
    print("Total time for fetching neighbors : ", t_end - t_init)
    return df


def agglo_clustering(data: pd.DataFrame, x_transformed):
    """
    Perform agglomerative clustering on the given data.

    Args:
        data (DataFrame): Input data.
        x_transformed (): Transformed input data array-like.

    Returns:
        DataFrame: DataFrame with cluster IDs and sizes.
    """
    dist_threshold = 3
    df = data.copy()

    t_init = time.time()
    # connectivety improves performance
    knn_graph = radius_neighbors_graph(
        x_transformed,
        radius=dist_threshold,
        mode="connectivity",
        metric="euclidean",
        include_self=False,
    )
    t_graph = time.time()
    print("Total time for clustering : ", t_graph - t_init)
    # Initialize AgglomerativeClustering
    aggcl = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=dist_threshold,
        linkage="ward",
        metric="euclidean",
        connectivity=knn_graph,
        compute_full_tree=True,
    )
    # Fit and predict cluster labels
    df["cluster_id"] = aggcl.fit_predict(x_transformed.toarray())
    t_pred = time.time()
    print("Total time for clustering : ", t_pred - t_graph)

    # Calculate cluster sizes
    df["cluster_size"] = df.groupby(["cluster_id"])["text"].transform("size")
    # Assign outliers to a separate cluster
    df.loc[df["cluster_size"] < 3, "cluster_id"] = 999999
    # Print number of clusters formed
    print("Total number of clusters formed: ", df["cluster_id"].nunique())
    t_clus = time.time()
    print("Total time for clustering : ", t_clus - t_init)

    return df


def tf_idf_agglo_clustering(_series: pd.Series) -> pd.DataFrame:
    """tf_idf_agglo_clustering _summary_

    _extended_summary_

    Args:
        _series (pd.Series): _description_

    Returns:
        _type_: _description_
    """

    unique_kw_df = _series.to_frame(name="text").drop_duplicates()
    print("total unique keywords for clustering: ", len(unique_kw_df))
    t_vec_init = time.time()

    # Determine the minimum document frequency for the vectorizer
    min_df_val = min(
        10, int(np.round(0.0005 * len(unique_kw_df))) if len(unique_kw_df) > 4000 else 1
    )

    # Initialize and fit the TF-IDF vectorizer
    vectorizer = TfidfVectorizer(analyzer=text_process_uni_bi, min_df=min_df_val)
    x_transformed = vectorizer.fit_transform(unique_kw_df["text"])

    t_vec = time.time()
    print("Time taken for vector creation : ", t_vec - t_vec_init)

    print("total unique keywords for clustering: ", len(unique_kw_df))
    # Perform agglomerative clustering
    df_clust = agglo_clustering(unique_kw_df, x_transformed)
    clustering_results = df_clust[["text", "cluster_id", "cluster_size"]]
    t_cluster = time.time()
    print("Time taken for cluster creation : ", t_cluster - t_vec)

    return clustering_results


def tf_idf_neighbors(_series: pd.Series) -> pd.DataFrame:
    """tf_idf_agglo_clustering _summary_

    _extended_summary_

    Args:
        _series (pd.Series): _description_

    Returns:
        _type_: _description_
    """

    unique_kw_df = _series.to_frame(name="text").drop_duplicates()
    print("total unique keywords for clustering: ", len(unique_kw_df))

    # Determine the minimum document frequency for the vectorizer
    min_df_val = min(
        10, int(np.round(0.0005 * len(unique_kw_df))) if len(unique_kw_df) > 4000 else 1
    )

    # Initialize and fit the TF-IDF vectorizer
    t_vec_init = time.time()
    vectorizer = TfidfVectorizer(analyzer=text_process_uni_bi, min_df=min_df_val)
    x_transformed = vectorizer.fit_transform(unique_kw_df["text"])

    t_vec = time.time()
    print("Time taken for vector creation : ", t_vec - t_vec_init)
    print("total unique keywords for finding neighbors: ", len(unique_kw_df))

    # Perform agglomerative clustering
    df_neighbors = NNGraph_fetch_neighbours(unique_kw_df, x_transformed)
    df_neighbors = df_neighbors[
        ["text", "neighbors", "neighbor_names", "neighbor_dist", "neighbor_count"]
    ]
    t_neigh = time.time()
    print("Time taken for neighbor creation : ", t_neigh - t_vec)

    return df_neighbors
