import pandas as pd
from datasets import load_dataset
import json 
import os 
import random


MAX_N_GENERATE = 1000 
N_TEST = 100           

def make_generate_test_from_hf(dataset_url, behavior_name="sycophancy", output_dir="./data"):
    """
    Creates generate and test JSON files from a Hugging Face JSON dataset.

    Args:
        dataset_url (str): URL to the JSON dataset.
        behavior_name (str): Name of the behavior (used for naming output files).
        output_dir (str): Directory to save the generated JSON files.
    """

    dataset = load_dataset("json", data_files=dataset_url, split="train", streaming=False)
    
    data = list(dataset)

    cleaned_data = []
    for item in data:
        question = item.get("question", "").replace("Question:", "").replace("Answer:", "").strip()
        answer_matching_behavior = item.get("answer_matching_behavior", "")
        answer_not_matching_behavior = item.get("answer_not_matching_behavior", "")

        # If any field is a list, just take the first element
        if isinstance(answer_matching_behavior, list):
            answer_matching_behavior = answer_matching_behavior[0]
        if isinstance(answer_not_matching_behavior, list):
            answer_not_matching_behavior = answer_not_matching_behavior[0]

        cleaned_data.append({
            "question": question,
            "answer_matching_behavior": answer_matching_behavior.strip(),
            "answer_not_matching_behavior": answer_not_matching_behavior.strip(),
        })

    random.shuffle(cleaned_data)

    n_generate = min(MAX_N_GENERATE, len(cleaned_data) - N_TEST)
    generate_data = cleaned_data[:n_generate]
    test_data = cleaned_data[-N_TEST:]

    os.makedirs(output_dir, exist_ok=True)
    generate_path = os.path.join(output_dir, f"{behavior_name}_generate.json")
    test_path = os.path.join(output_dir, f"{behavior_name}_test.json")
    open_ended_test_path = os.path.join(output_dir, f"{behavior_name}_open_ended_test.json")

    with open(generate_path, "w") as f:
        json.dump(generate_data, f, indent=4)
    with open(test_path, "w") as f:
        json.dump(test_data, f, indent=4)

    open_ended_test = [{"question": item["question"].split("\n")[0].strip()} for item in test_data]
    with open(open_ended_test_path, "w") as f:
        json.dump(open_ended_test, f, indent=4)

    print(f"{behavior_name}: n_generate: {n_generate} | n_test: {N_TEST}")
    return generate_path, test_path, open_ended_test_path


def process_raw_json_datasets(file_path):
    """
    Reads a JSON dataset and converts it into a DataFrame with prompts,
    sycophancy labels, and pair IDs for contrastive examples.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        pd.DataFrame: DataFrame with columns ['pair_id', 'prompt', 'is_syco'].
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    records = []

    for idx, item in enumerate(data):
        pair_id = idx  

        question = item.get("question", "").replace("Question:", "").strip()
        answer_matching = item.get("answer_matching_behavior", "")
        answer_not_matching = item.get("answer_not_matching_behavior", "")

        if isinstance(answer_matching, list):
            answer_matching = answer_matching[0]
        if isinstance(answer_not_matching, list):
            answer_not_matching = answer_not_matching[0]

        prompt_syco = f"Question: {question}\nAnswer: {answer_matching}"
        records.append({"pair_id": pair_id, "prompt": prompt_syco, "is_syco": 1})

        prompt_non_syco = f"Question: {question}\nAnswer: {answer_not_matching}"
        records.append({"pair_id": pair_id, "prompt": prompt_non_syco, "is_syco": 0})

    df = pd.DataFrame(records)
    return df