ST_BERT_CONFIG = {
    "URL": {
        "MiniLML6V2": "https://hrmn5k2173.execute-api.ap-south-1.amazonaws.com/default/SentenceTransformersMiniLML6V2",
        "BertBaseMeanTokens": (
            "https://p8uohxj3u6.execute-api.ap-south-1.amazonaws.com/default/SentenceTransformerBertBaseMeanTokens"
        ),
        "BERTMultiLingualMiniL12Baseten": "https://app.baseten.co/model_versions/womkr7q/predict",
    },
    "RETRY_COUNT": 3,
    "RETRY_DELAY": 5,
    "DTYPE": np.float32,
}
