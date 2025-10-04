import asyncio
import json
import time
import aiohttp
import numpy as np
from .utils import chunks


async def bert_api_call_async_make_calls(
    session: any, index: int, sentences: list[str], model: str, ST_BERT_CONFIG: dict
) -> tuple[int, list]:
    """_summary_

    Args:
        session (any): Session
        index (int): Session index
        sentences (list[str]): List of keywords
        model (str): Model to use
        ST_BERT_CONFIG (dict): Config dict

    Raises:
        Exception: _description_
        Exception: _description_

    Returns:
        tuple[int, list]: List of index and embeddings corresponding to the passed keywords
    """

    url = ST_BERT_CONFIG["URL"][model]
    RETRY_COUNT = ST_BERT_CONFIG["RETRY_COUNT"]
    RETRY_DELAY = ST_BERT_CONFIG["RETRY_DELAY"]
    DTYPE = ST_BERT_CONFIG["DTYPE"]
    headers = {"Content-Type": "application/json"}

    payload = json.dumps({"doc": sentences})
    resp = None
    embedding = None

    for _iter in range(1 + RETRY_COUNT):
        response = None
        try:
            async with session.post(url, headers=headers, data=payload, timeout=300) as resp:
                response = await resp.json()

                embedding = [np.asarray(r, dtype=DTYPE) for r in response]

                if len(sentences) == len(embedding):
                    return index, embedding
                else:
                    raise Exception(
                        f"Error with SentenceTransformer API! Error Message: {str(response.get('message',''))}!!"
                    )
        except Exception as e:
            print(
                "Error while fetching embedding using sentence transformer api: ",
                str(e),
            )
            print("Inputs : ", sentences)
            print("API Response : %s" % str(response))
            print(f"Index: {index}")
        print(f"Retrying {_iter+1} in {RETRY_DELAY} seconds!!")
        time.sleep(RETRY_DELAY)
    raise Exception("Error while fetching sentence transformer embeddings!!")


async def bert_api_call_async_gather_calls(sentences_list: list[str], model: str, ST_BERT_CONFIG: dict) -> list:
    """Gather async calls for embeddings

    Args:
        sentences_list (list[str]): List of keywords
        model (str): Model name
        ST_BERT_CONFIG (dict): Model Config

    Returns:
        list: List of embeddings
    """
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(force_close=True, limit=3)) as session:
        tasks = []
        for index, sentences in enumerate(sentences_list):
            embeddings_out = bert_api_call_async_make_calls(session, index, sentences, model, ST_BERT_CONFIG)
            tasks.append(asyncio.ensure_future(embeddings_out))
        out = await asyncio.gather(*tasks)
        return out


def bert_api_async(
    keywords: list[str],
    ST_BERT_CONFIG: dict,
    model: str = "SentenceTransformersMiniLML6V2",
    chunk_size: int = 200,
) -> list:
    """Fetch embeddings for the passed keywords in async mode

    Args:
        keywords (list[str]): List of keywords
        ST_BERT_CONFIG (dict): Config dict
        model (str, optional): Model to use. Defaults to "SentenceTransformersMiniLML6V2".
        chunk_size (int, optional): Chunk size for async mode. Defaults to 200.

    Returns:
        list: List of embeddings corresponding to the passed keywords
    """
    all_embeddings = {}
    keyword_chunks = chunks(keywords, chunk_size)

    out = asyncio.run(bert_api_call_async_gather_calls(keyword_chunks, model, ST_BERT_CONFIG))
    for chunk_iter in out:
        iter = chunk_iter[0]
        embedding_iter = chunk_iter[1]
        for i, keyword in enumerate(keyword_chunks[iter]):
            all_embeddings[keyword] = embedding_iter[i]

    return list(all_embeddings.values())
