import re
from .gpt3_calls_cai import (
    get_multiple_gpt_call_response,
)
from .utils import chunks

MAX_TARGETINGS_PER_COHORT = 30
MIN_KEYWORDS_FOR_COHORT_NAME = 3
EXTRACT_CLUSTER_NAMES_PARAMETERS = {
    "model": "gpt-3.5-turbo-instruct",
    "temperature": 0.75,
    "max_tokens": 300,
    "top_p": 1,
    "frequency_penalty": 0.8,
    "presence_penalty": 0.4,
    "stop": ["###"],
    "n": 3,
}
EXTRACT_CLUSTER_NAMES_DETAILED_PROMPT = """Recommend a unique descriptive 2-3 word cluster name for the passed clusters of google search keywords. 
Ensure 'Cluster ID: Cluster name' format for all clusters with clusters separated by new line. Also given that the text is of search phrases type it might not always follow proper grammer.
Here are some examples :
###
Query:
```
Recomend a unique descriptive 2-3 word cluster name for the following clusters of google search keywords:
1: best place to sell ring,best place to buy rings online,order ring online,best site to buy rings online,where to buy ring settings only,best place to finance a ring
2: david yurman earrings with diamonds,david yurman bracelet with diamonds,david yurman silver bracelet with diamonds
3: september ring stone,december mothers ring,september birth ring
4: buy real gold chains online,best place to buy gold chains online,best websites to buy real gold chains,best website to buy gold chains,best site to buy gold chains,best place to buy real gold chains online
5: cartier ring women,a cartier ring,cartier basic ring,cartier ring cartier,buy cartier ring,cartier ring for her
6: women cartier ring price,cartier ring price,cartier marriage ring price,gra ring price
7: unique engagement settings,three stone engagement settings,3 stone engagement settings,engagement settings no stone local,engagement settings for sale
8: mood ring colors,mood mood ring colors,mood colors for mood rings,mood ring color descriptions,mood ring color key,emotion mood ring colors,all mood ring colors,all the colors of a mood ring,you re in the mood ring color chart,colours of mood rings chart
9: 14k stone ring,settings for three stone rings,three stone ring settings only,women ring with stone,3 stone halo ring settings,3 stone ring mountings,unique 3 stone ring settings,three stone ring mount,3 stone mountings
10: how are diamonds shaped,how are diamonds man made,are diamonds natural or manmade,diamonds how to buy,are man made diamonds any good
11: david yurman necklace with diamonds,david yurman diamond necklace,david yurman pendant necklace with diamonds,david yurman diamond pendant,david yurman diamond pendant necklace,david yurman a pendant with diamonds in gold on chain,david yurman stone necklace
12: blue nile diamonds near me,blue nile diamond ring price,blue nile anniversary rings,blue nile diamond settings,blue nile halo,blue nile halo ring,diamond anniversary rings blue nile
13: custom engagement,build your own engagement,nontraditional engagement,create your own engagement,design your own engagement,design engagement,debeers engagement,design your own engagement setting
14: pdf printable ring sizer strip,woman printable ring sizer strip,free printable ring sizer strip,downloadable pdf printable ring sizer strip
```
> Results:
1: Online Ring Shopping
2: David Yurman Jewelry
3: Birthstone Rings
4: Gold Chain Shopping
5: Cartier Rings
6: Cartier Ring Price Inquiry
7: Engagement Ring Settings
8: Mood Ring Colors
9: Three Stone Settings
10: Diamond Buying Guide
11: David Yurman Necklaces
12: Blue Nile Rings
13: Custom Engagement Design
14: Printable Ring Sizer
###
Query:
```
Recomend a unique descriptive 2-3 word cluster name for the following clusters of google search keywords:
{targetings_string}
```
"""


def parse_result(result, allowed_keys: set[str] | None = None) -> dict[str, str]:
    if allowed_keys is None:
        allowed_keys = set()

    link_dict = {}
    all_lines = result.lower().split("\n")
    for line in all_lines:
        key = line.split(":")[0]
        value = line.split(":")[-1]
        key = re.sub("[^0-9]", "", key).strip()
        value = re.sub("[^a-z ]", "", value).strip()
        if (len(key) > 0) and (len(value) >= MIN_KEYWORDS_FOR_COHORT_NAME) and (key in allowed_keys):
            link_dict[key] = value

    return link_dict


def format_results(results, cluster_order):
    res_dict = {}
    out_dict = {}

    for result in results:
        parsed_dict = parse_result(result, allowed_keys=set(cluster_order.keys()))
        for res_id in parsed_dict:
            if res_id in res_dict:
                res_dict[res_id].add(parsed_dict[res_id])
            else:
                res_dict[res_id] = set([parsed_dict[res_id]])

    for res_id in res_dict:
        if res_id in cluster_order:
            if cluster_order[res_id] in out_dict:
                out_dict[cluster_order[res_id]].update(res_dict[res_id])
            else:
                out_dict[cluster_order[res_id]] = set(res_dict[res_id])

    return out_dict


def create_tar_strings(targetings_list):
    targetings_string = ""
    for i in range(len(targetings_list)):
        targetings_string = targetings_string + str(i + 1) + ": " + ",".join(targetings_list[i]) + "\n"
    return targetings_string


def extract_cluster_names(all_cluster_dict, chunk_size=10):
    all_cluster_names = {}
    all_cluster_dict = {str(k): all_cluster_dict[k] for k in all_cluster_dict}
    cluster_ids = list(all_cluster_dict.keys())
    clusters_ids_chunks = chunks(cluster_ids, chunk_size)

    iter_info = []
    for iter_id, ids_chunk in enumerate(clusters_ids_chunks):
        iter_info_dict = {
            "id": iter_id,
            "clus_ids_chunk": ids_chunk,
            "targetings_strings": create_tar_strings(
                [all_cluster_dict[c][:MAX_TARGETINGS_PER_COHORT] for c in ids_chunk]
            ),
            "cluster_order_dict": {str(i + 1): id for i, id in enumerate(ids_chunk)},
        }
        iter_info_dict["gpt_call_payload"] = {
            "model": EXTRACT_CLUSTER_NAMES_PARAMETERS["model"],
            "prompt": EXTRACT_CLUSTER_NAMES_DETAILED_PROMPT.format(
                targetings_string=iter_info_dict["targetings_strings"]
            ),
            "parameters": EXTRACT_CLUSTER_NAMES_PARAMETERS,
        }

        iter_info.append(iter_info_dict)

    gpt_payloads = [iter_info[i]["gpt_call_payload"] for i in range(len(iter_info))]
    gpt_raw_response = get_multiple_gpt_call_response(gpt_payloads)

    for i in range(len(iter_info)):
        iter_info[i]["gpt_response"] = gpt_raw_response[i]
        iter_info[i]["cluster_names_chunk"] = format_results(
            results=iter_info[i]["gpt_response"], cluster_order=iter_info[i]["cluster_order_dict"]
        )

        for c in iter_info[i]["cluster_names_chunk"]:
            relevant_names = iter_info[i]["cluster_names_chunk"][c]
            if len(relevant_names) == 0:
                continue
            if c in all_cluster_names:
                all_cluster_names[c].update(relevant_names)
            else:
                all_cluster_names[c] = relevant_names

    for c in all_cluster_names:
        all_cluster_names[c] = list(set(all_cluster_names[c]))
    print(
        "Found no cluster name using gpt3 for these cluster_ids : ",
        list(set(all_cluster_dict.keys()) - set(all_cluster_names.keys())),
    )
    return all_cluster_names
