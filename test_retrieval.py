import asyncio
import aiohttp
import os
import json
from tqdm import tqdm
from loguru import logger
logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""))

RETRIEVAL_ENDPOINT = "http://0.0.0.0:5100/search"
HEADERS = {"Content-Type": "application/json"}
DATA_FOLDER = "generated_data"
NUM_BLOCKS_TO_SEARCH = 20
TOP_K_TO_EVALUATE = [1, 3, 10, 20]

def preprocess_multiple_choice_query(query: str):
    start_idx = query.find("[Input Question]\n")
    end_idx = -9 # Exclude the "\nAnswer: " part
    return query[start_idx + len("[Input Question]\n") : end_idx]

async def search(query: str, num_blocks: int = 1):
    data = {
        "query": [f"{query}"],
        "num_blocks": num_blocks,
        "languages": ["en"]
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(RETRIEVAL_ENDPOINT, headers=HEADERS, json=data) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception(f"Failed to retrieve data from {RETRIEVAL_ENDPOINT}")

async def compute_accuracy():
    results = {}
    results["summary"] = {}

    for file in sorted(os.listdir(DATA_FOLDER), reverse=True):
        if file.endswith(".jsonl"):
            task_name = "_".join(file.split("_")[:-1])
            results_for_task = []

            logger.info(f"========== Processing {task_name}... ==========")
            with open(os.path.join(DATA_FOLDER, file), "r") as f:
                for line in tqdm(f, total=100):
                    data = json.loads(line)

                    # Load query and preprocess if task is multi_choice
                    query = data["query"]
                    if task_name == "multi_choice":
                        query = preprocess_multiple_choice_query(query)
                    # Load the dataset_entry
                    dataset_entry = data["dataset_entry"]
                    gt_title = dataset_entry["title"].strip()
                    gt_topic = dataset_entry["topic"].strip()

                    # Search the query in the WikiChat index
                    search_result = await search(query, num_blocks=NUM_BLOCKS_TO_SEARCH)
                    search_result = search_result[0] # Only one query
                    search_result_titles = [title.strip() for title in search_result["title"]]
                    search_result_sections = [[section.strip() for section in search_result["full_section_title"][i].split(">")[1:]] for i in range(NUM_BLOCKS_TO_SEARCH)]

                    # Compute the results for the query
                    results_for_query = {}
                    for k in TOP_K_TO_EVALUATE:
                        results_for_query[f"title_match_top_{k}"] = int(gt_title in search_result_titles[:k])
                        results_for_query[f"section_match_top_{k}"] = int(any(gt_topic in section for section in search_result_sections[:k]))

                    # Add the results for the query to the results for the task
                    results_for_task.append(results_for_query)

            results[task_name] = results_for_task
            # Summary results for each task
            results["summary"][task_name] = {}
            for k in TOP_K_TO_EVALUATE:
                results["summary"][task_name][f"avg_title_match_top_{k}"] = sum(result[f"title_match_top_{k}"] for result in results_for_task) / len(results_for_task)
                results["summary"][task_name][f"avg_section_match_top_{k}"] = sum(result[f"section_match_top_{k}"] for result in results_for_task) / len(results_for_task)
    
    with open("generated_data/results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    asyncio.run(compute_accuracy())
