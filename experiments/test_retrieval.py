import asyncio
import aiohttp
import os
import json
from tqdm import tqdm
from typing import ClassVar, List, Tuple
from pydantic import BaseModel
import wikipedia
import sys
import random
from urllib.parse import quote
from bs4 import BeautifulSoup
from loguru import logger
logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""))

import spacy
nlp = spacy.load('en_core_web_sm')
from rapidfuzz import fuzz

RETRIEVAL_ENDPOINT = "http://0.0.0.0:5100/search"
HEADERS = {"Content-Type": "application/json"}
DATA_FOLDER = "experiments/generated_data"
NUM_BLOCKS_TO_SEARCH = 20
TOP_K_TO_EVALUATE = [1, 3, 10, 20]
EXCLUDE_HEADERS = ("See also", "References", "Further reading", "External links")
EXCLUDE_CATEGORIES = ("articles", "wiki", "pages", "cs1")
min_length_words: int = 0


###### Wikipedia utility functions ######
async def _get_page(
    title: str, pageid: str | None = None, auto_suggest: bool = False, redirect: bool = True, seed: int | None = None
) -> wikipedia.WikipediaPage:
    """Cached Wikipedia page loading."""
    try:
        # Note: wikipedia.page is synchronous, but it's mainly used for metadata
        page = wikipedia.page(title=title, pageid=pageid, auto_suggest=auto_suggest, redirect=redirect)
        return page

    except wikipedia.DisambiguationError as e:
        logger.debug(f"{e.__class__.__name__} loading page {title!r}: {e}")
        pages = sys.exc_info()[1].args[1]
        if not isinstance(pages, list):
            return None
        title = random.Random(seed).choice(pages)
        return await _get_page(title, auto_suggest=auto_suggest, redirect=redirect)

    except wikipedia.PageError as e:
        logger.warning(f"{e.__class__.__name__} loading page {title!r}: {e}")
        if not auto_suggest:
            return await _get_page(title, auto_suggest=True, redirect=redirect)
        return None
    
async def get_article_sections(title: str, session: aiohttp.ClientSession) -> dict[str, str]:
    # Replace special characters in the title
    title = replace_special_characters(title)

    # Fetch the HTML content of the Wikipedia article
    url = f"https://en.wikipedia.org/wiki/{title}"
    async with session.get(url) as response:
        html_content = await response.text()

    # Parse the HTML using BeautifulSoup
    soup = BeautifulSoup(html_content, "html.parser")

    sections = {}
    for section in soup.find_all("h2"):
        if (p_tag := section.find_next("p")) is not None:
            sections[section.text] = p_tag.text

    return sections

async def process_page(
    page: wikipedia.WikipediaPage, session: aiohttp.ClientSession,
    exclude_sections: list | None = None, valid_section: callable = None
) -> list[tuple[str, str]]:
    
    title = page.title
    sections = await get_article_sections(title, session)

    if exclude_sections:
        sections = {k: v for k, v in sections.items() if k not in exclude_sections}

    valid_sections = [        
        (key, value) for key, value in sections.items() if not valid_section or valid_section(sections[key])
    ]

    return valid_sections if valid_sections else None


async def get_article_sections_with_contents(title: str, session: aiohttp.ClientSession) -> list[tuple[str, str]]:
    page = await _get_page(title=title)
    if page is None:
        return None
    
    # Only return a sections with a minimum number of words
    selected_section = await process_page(
        page,
        session=session,
        exclude_sections=EXCLUDE_HEADERS,
        valid_section=lambda x: len(x.split()) >= min_length_words,
    )
    if not selected_section:
        logger.warning(f"No valid sections found for {page.title}")
        return None

    return selected_section

async def get_best_section_match(search_result_title: str, search_result_content: str,
                                 similarity_method: str,
                                 session: aiohttp.ClientSession) -> str:
    extracted_sections_contents = await get_article_sections_with_contents(search_result_title, session)
    if not extracted_sections_contents:
        return None
    best_section_match = max(extracted_sections_contents, key=lambda x: compute_similarity(x[1], search_result_content, similarity_method))[0]
    return search_result_title + " > " + best_section_match


# Similarity functions
def compute_similarity(retrieved_context: str, extracted_context: str,
                       method: str = "spacy"):
    if method == "spacy":
        doc1 = nlp(retrieved_context)
        doc2 = nlp(extracted_context)
        return doc1.similarity(doc2)
    elif method == "rapidfuzz":
        return fuzz.ratio(retrieved_context, extracted_context)


###### Refinement functions ######
def preprocess_multiple_choice_query(query: str):
    start_idx = query.find("[Input Question]\n")
    end_idx = -9 # Exclude the "\nAnswer: " part
    return query[start_idx + len("[Input Question]\n") : end_idx]

def replace_special_characters(page_title: str):
    return quote(page_title)

###### Search API ######
async def search(query: str, num_blocks: int = 1, session: aiohttp.ClientSession = None):
    data = {
        "query": [f"{query}"],
        "num_blocks": num_blocks,
        "languages": ["en"]
    }
    async with session.post(RETRIEVAL_ENDPOINT, headers=HEADERS, json=data) as response:
        if response.status == 200:
            return await response.json()
        else:
            raise Exception(f"Failed to retrieve data from {RETRIEVAL_ENDPOINT}")


###### Main function ######
async def compute_accuracy(output_file: str, method: str):
    logger.info(f"Computing accuracy with method {method}...")
    results = {}
    results["summary"] = {}
    async with aiohttp.ClientSession() as session:
        for file in sorted(os.listdir(DATA_FOLDER), reverse=True):
            if file.endswith(".jsonl"):
                task_name = "_".join(file.split("_")[:-1])
                results_for_task = []

                logger.info(f"========== Processing {task_name}... ==========")
                with open(os.path.join(DATA_FOLDER, file), "r") as f:
                    for line in tqdm(f, total=100):
                        data = json.loads(line)

                        # Load query and preprocess if task is multi_choice
                        # Can modify as we want
                        query = data["query"]
                        if task_name == "multi_choice":
                            query = preprocess_multiple_choice_query(query)
                        
                        # Load the dataset_entry
                        # DO NOT TOUCH the dataset_entry, it's the ground truth
                        dataset_entry = data["dataset_entry"]
                        gt_title = dataset_entry["title"].strip()
                        gt_topic = gt_title + " > " + dataset_entry["topic"].strip()
                        gt_content = dataset_entry["content"].strip()

                        # Search the query in the WikiChat index
                        search_result = await search(query, num_blocks=NUM_BLOCKS_TO_SEARCH, session=session)
                        search_result = search_result[0] # Only one query
                        search_result_titles = [title.strip() for title in search_result["title"]]
                        search_result_contents = [content.strip() for content in search_result["text"]]

                        search_result_best_section_match = await asyncio.gather(
                            *[get_best_section_match(title, content, method, session) for title, content in zip(search_result_titles, search_result_contents)]
                        )

                        # Compute the results for the query
                        results_for_query = {}
                        for k in TOP_K_TO_EVALUATE:
                            results_for_query[f"title_match_top_{k}"] = int(gt_title in search_result_titles[:k])
                            results_for_query[f"section_match_top_{k}"] = int(gt_topic in search_result_best_section_match[:k])
                        results_for_query["query"] = query
                        results_for_query["gt_title"] = gt_title
                        results_for_query["gt_topic"] = gt_topic
                        results_for_query["gt_content"] = gt_content
                        results_for_query["search_contents"] = search_result_contents
                        results_for_query["extracted_sections"] = search_result_best_section_match

                        # Add the results for the query to the results for the task
                        results_for_task.append(results_for_query)

                results[task_name] = results_for_task
                # Summary results for each task
                results["summary"][task_name] = {}
                for k in TOP_K_TO_EVALUATE:
                    results["summary"][task_name][f"avg_title_match_top_{k}"] = sum(result[f"title_match_top_{k}"] for result in results_for_task) / len(results_for_task)
                    results["summary"][task_name][f"avg_section_match_top_{k}"] = sum(result[f"section_match_top_{k}"] for result in results_for_task) / len(results_for_task)

    with open(f"experiments/results/{output_file}", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-file", type=str, default="results.json")
    parser.add_argument("--method", type=str, default="rapidfuzz")
    args = parser.parse_args()
    asyncio.run(compute_accuracy(args.output_file, args.method))
