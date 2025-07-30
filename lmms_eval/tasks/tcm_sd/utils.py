import json
import random
import re
from collections import defaultdict

import numpy as np
from loguru import logger as eval_logger

MULTI_CHOICE_PROMPT = "请只回答选项字母（如：A）。"
DIRECT_PROMPT = "请直接回答证候名称，不需要解释。"

def parse_options(options):
    """Parse options into formatted string with A, B, C, D, E format"""
    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    choices_str = "\n".join([f"{option_letter}. {option}" for option_letter, option in zip(option_letters, options)])
    return choices_str


def tcm_sd_doc_to_text_multiple_choice(doc, lmms_eval_specific_kwargs=None):
    """Convert document to text prompt for TCM multiple choice questions"""
    prompt = doc["prompt"]
    full_prompt = f"{prompt}"
    return full_prompt


def tcm_sd_doc_to_text_direct(doc, lmms_eval_specific_kwargs=None):
    """Convert document to text prompt for TCM direct diagnosis"""
    prompt = doc["prompt"]
    full_prompt = f"{prompt}"
    return full_prompt


def tcm_sd_doc_to_text_rc_five(doc, lmms_eval_specific_kwargs=None):
    """Convert document to text prompt for TCM reading comprehension (five options)"""
    prompt = doc["prompt"]
    full_prompt = f"{prompt}"
    return full_prompt


def tcm_sd_doc_to_text_rc_all(doc, lmms_eval_specific_kwargs=None):
    """Convert document to text prompt for TCM reading comprehension (all options)"""
    prompt = doc["prompt"]
    full_prompt = f"{prompt}"
    return full_prompt


def get_multi_choice_info(options):
    """
    Given the list of options for multiple choice question
    Return the index2ans and all_choices
    """
    start_chr = "A"
    all_choices = []
    index2ans = {}
    for i, option in enumerate(options):
        index2ans[chr(ord(start_chr) + i)] = option
        all_choices.append(chr(ord(start_chr) + i))
    return index2ans, all_choices


def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D, E.
    """
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "

    index_ans = True
    ans_with_brack = False
    candidates = []
    
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice} " in response:
                candidates.append(choice)

    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice}." in response:
                candidates.append(choice)

    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False

    if len(candidates) == 0:
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    start_indexes.append(response.rfind(f"({can})"))
            else:
                for can in candidates:
                    start_indexes.append(response.rfind(f" {can} "))
        else:
            for can in candidates:
                start_indexes.append(response.lower().rfind(index2ans[can].lower()))
        pred_index = candidates[np.argmax(start_indexes)]
    else:
        pred_index = candidates[0]
    return pred_index


def tcm_sd_process_results_direct(doc, results):
    """Process results for TCM direct diagnosis"""
    from lmms_eval.api.metrics import exact_match_hf_evaluate
    
    result = results[0]
    
    # Handle different response formats
    if hasattr(result, 'doc_ids'):
        # ResponseWithDocIds object
        pred = str(result).strip()
        retrieved_doc_ids = result.doc_ids
    elif isinstance(result, dict):
        # Legacy dict format
        pred = result["text"].strip()
        retrieved_doc_ids = result.get("doc_ids", [])
    else:
        # String format
        pred = result.strip()
        retrieved_doc_ids = []
    
    # Calculate exact_match using built-in function
    exact_match_result = exact_match_hf_evaluate(
        predictions=[pred],
        references=[doc["expected_answer"]],
        ignore_case=True,
        ignore_punctuation=True,
        ignore_numbers=True
    )
    
    expected_doc_id = doc.get("expected_doc_id")
    hit = expected_doc_id in retrieved_doc_ids if expected_doc_id is not None and retrieved_doc_ids else False
    
    return {
        "exact_match": exact_match_result["exact_match"],
        "hit_rate": {
            "user_id": doc["user_id"],
            "expected_doc_id": expected_doc_id,
            "retrieved_doc_ids": retrieved_doc_ids,
            "hit": hit
        },
        "submission": {doc["user_id"]: pred}
    }


def tcm_sd_process_results_multiple_choice(doc, results):
    """Process results for TCM multiple choice questions"""
    from lmms_eval.api.metrics import exact_match_hf_evaluate
    
    result = results[0]
    
    # Handle different response formats
    if hasattr(result, 'doc_ids'):
        # ResponseWithDocIds object
        pred = str(result).strip()
        retrieved_doc_ids = result.doc_ids
    elif isinstance(result, dict):
        # Legacy dict format
        pred = result["text"].strip()
        retrieved_doc_ids = result.get("doc_ids", [])
    else:
        # String format
        pred = result.strip()
        retrieved_doc_ids = []
        
    options = doc["options"]
    index2ans, all_choices = get_multi_choice_info(options)
    parsed_pred = parse_multi_choice_response(pred, all_choices, index2ans)
    
    # Calculate exact_match using built-in function
    exact_match_result = exact_match_hf_evaluate(
        predictions=[parsed_pred],
        references=[doc["expected_answer"]],
        ignore_case=True,
        ignore_punctuation=True,
        ignore_numbers=True
    )
    
    expected_doc_id = doc.get("expected_doc_id")
    hit = expected_doc_id in retrieved_doc_ids if expected_doc_id is not None and retrieved_doc_ids else False
        
    return {
        "exact_match": exact_match_result["exact_match"],
        "hit_rate": {
            "user_id": doc["user_id"],
            "expected_doc_id": expected_doc_id,
            "retrieved_doc_ids": retrieved_doc_ids,
            "hit": hit
        },
        "submission": {doc["user_id"]: parsed_pred}
    }


def tcm_sd_process_results_rc_five(doc, results):
    """Process results for TCM reading comprehension (five options)"""
    from lmms_eval.api.metrics import exact_match_hf_evaluate
    
    result = results[0]
    
    # Handle different response formats
    if hasattr(result, 'doc_ids'):
        # ResponseWithDocIds object
        pred = str(result).strip()
        retrieved_doc_ids = result.doc_ids
    elif isinstance(result, dict):
        # Legacy dict format
        pred = result["text"].strip()
        retrieved_doc_ids = result.get("doc_ids", [])
    else:
        # String format
        pred = result.strip()
        retrieved_doc_ids = []
    
    # Calculate exact_match using built-in function
    exact_match_result = exact_match_hf_evaluate(
        predictions=[pred],
        references=[doc["expected_answer"]],
        ignore_case=True,
        ignore_punctuation=True,
        ignore_numbers=True
    )
    
    expected_doc_id = doc.get("expected_doc_id")
    hit = expected_doc_id in retrieved_doc_ids if expected_doc_id is not None and retrieved_doc_ids else False
    
    return {
        "exact_match": exact_match_result["exact_match"],
        "hit_rate": {
            "user_id": doc["user_id"],
            "expected_doc_id": expected_doc_id,
            "retrieved_doc_ids": retrieved_doc_ids,
            "hit": hit
        },
        "submission": {doc["user_id"]: pred}
    }


def tcm_sd_process_results_rc_all(doc, results):
    """Process results for TCM reading comprehension (all options)"""
    from lmms_eval.api.metrics import exact_match_hf_evaluate
    
    result = results[0]
    
    # Handle different response formats
    if hasattr(result, 'doc_ids'):
        # ResponseWithDocIds object
        pred = str(result).strip()
        retrieved_doc_ids = result.doc_ids
    elif isinstance(result, dict):
        # Legacy dict format
        pred = result["text"].strip()
        retrieved_doc_ids = result.get("doc_ids", [])
    else:
        # String format
        pred = result.strip()
        retrieved_doc_ids = []
    
    # Calculate exact_match using built-in function
    exact_match_result = exact_match_hf_evaluate(
        predictions=[pred],
        references=[doc["expected_answer"]],
        ignore_case=True,
        ignore_punctuation=True,
        ignore_numbers=True
    )
    
    expected_doc_id = doc.get("expected_doc_id")
    hit = expected_doc_id in retrieved_doc_ids if expected_doc_id is not None and retrieved_doc_ids else False
    
    return {
        "exact_match": exact_match_result["exact_match"],
        "hit_rate": {
            "user_id": doc["user_id"],
            "expected_doc_id": expected_doc_id,
            "retrieved_doc_ids": retrieved_doc_ids,
            "hit": hit
        },
        "submission": {doc["user_id"]: pred}
    }


def eval_multi_choice(gold_i, pred_i):
    """Evaluate a multiple choice instance."""
    return gold_i == pred_i


def evaluate_tcm_sd_hit_rate(samples):
    """Batch evaluation for TCM hit rate"""
    total_hits = 0
    total_samples = 0
    
    for sample in samples:
        if sample.get("expected_doc_id") is not None:  # Only count samples with expected_doc_id
            total_samples += 1
            if sample.get("hit", False):
                total_hits += 1
    
    return {"hit_rate": total_hits / total_samples if total_samples > 0 else 0}


def tcm_sd_aggregate_results_hit_rate(results):
    """Aggregate hit rate results for TCM diagnosis"""
    metric_dict = evaluate_tcm_sd_hit_rate(results)
    total_samples = len([r for r in results if r.get("expected_doc_id") is not None])
    hit_rate = metric_dict["hit_rate"]
    
    eval_logger.info(f"TCM SD Hit Rate Evaluation Results:")
    eval_logger.info(f"Total samples with expected_doc_id: {total_samples}")
    eval_logger.info(f"Hit Rate: {hit_rate:.4f}")
    
    return hit_rate 
