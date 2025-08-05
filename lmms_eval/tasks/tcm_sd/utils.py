import json
import random
import re
from collections import defaultdict

import numpy as np
from loguru import logger as eval_logger

MULTI_CHOICE_PROMPT = "请只回答选项字母（如：A）。"
DIRECT_PROMPT = "请直接回答证候名称，不需要解释。"

# CoT filtering keywords for Chinese TCM responses
TCM_ANSWER_INDICATORS = [
    "答案是", "答案:", "答案：", "答案为", "答案为：",
    "因此", "所以", "最终", "综上", "总结",
    "诊断为", "诊断是", "诊断:", "诊断：",
    "证候为", "证候是", "证候:", "证候：",
    "证候名称", "证候名称:", "证候名称：", "证候名称为",
    "最符合的中医证候", "最符合的证候", "最符合的中医证候为", "最符合的证候为",
    "最符合的证候名称", "最符合的证候名称为", "最符合的证候名称:",
    "最后", "最终答案", "最终诊断",
    "结论", "结论是", "结论:", "结论：",
    "判断为", "判断是", "考虑为",
    "选择", "选", "应该是", "应该选",
    # English keywords for mixed responses
    "answer is", "answer:", "therefore", "so",
    "final", "finally", "conclusion", "diagnosis",
    "the answer", "result", "thus"
]

def filter_cot_response(response, answer_type="direct"):
    """
    Filter Chain-of-Thought reasoning from response to extract the final answer.
    
    Args:
        response (str): The model's response that may contain CoT reasoning
        answer_type (str): Type of answer expected - "direct" or "multiple_choice"
    
    Returns:
        str: Filtered response with CoT reasoning removed
    """
    if not response or not response.strip():
        return response
    
    original_response = response
    response = response.strip()
    
    # Strategy 1: Look for structured answer patterns
    # Handle "Answer:" patterns and TCM-specific patterns
    answer_patterns = [
        # Basic answer patterns
        r'答案[：:]\s*(.+?)(?:\n|$)',
        r'诊断[：:]\s*(.+?)(?:\n|$)', 
        r'证候[：:]\s*(.+?)(?:\n|$)',
        r'结论[：:]\s*(.+?)(?:\n|$)',
        r'answer[：:]\s*(.+?)(?:\n|$)',
        r'diagnosis[：:]\s*(.+?)(?:\n|$)',
        
        # TCM-specific patterns
        r'证候名称[：:]\s*(.+?)(?:\n|$)',
        r'证候名称为[：:]?\s*(.+?)(?:\n|$)',
        r'最符合的.*?证候.*?为[：:]?\s*["""]?(.+?)["""]?(?:[。．\n]|$)',
        r'最符合的.*?证候[：:]?\s*["""]?(.+?)["""]?(?:[。．\n]|$)',
        r'证候名称为[：:]?\s*["""]?(.+?)["""]?(?:[。．\n]|$)',
        
        # Pattern for quoted answers
        r'["""]([^"""]+)["""](?:[。．]?$)',
        
        # Pattern for answers ending with period
        r'(?:因此|所以|综上|最终|最后).*?[：:]?\s*(.+?)(?:[。．]|$)',
    ]
    
    for pattern in answer_patterns:
        matches = re.findall(pattern, response, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        if matches:
            # Take the last match (most likely to be the final answer)
            extracted = matches[-1].strip()
            # Clean up common characters
            extracted = re.sub(r'^[：:：，,。．\s]+', '', extracted)
            extracted = re.sub(r'[。．\s]+$', '', extracted)
            # Remove asterisks and formatting
            extracted = re.sub(r'\*+', '', extracted)
            if extracted and not extracted.isspace() and len(extracted) <= 50:
                return extracted
    
    # Strategy 2: Keep the last answer pattern
    # Find the last occurrence of key indicators
    last_indicator_pos = -1
    last_indicator = ""
    
    for indicator in TCM_ANSWER_INDICATORS:
        pos = response.lower().rfind(indicator.lower())
        if pos > last_indicator_pos:
            last_indicator_pos = pos
            last_indicator = indicator
    
    if last_indicator_pos != -1:
        # Extract content after the last indicator
        after_indicator = response[last_indicator_pos + len(last_indicator):].strip()
        
        # Clean up common prefixes and separators
        for sep in ["：", ":", "是", "为", "："]:
            if after_indicator.startswith(sep):
                after_indicator = after_indicator[len(sep):].strip()
                break
        
        # Handle quoted content
        quote_match = re.search(r'^["""]([^"""]+)["""]', after_indicator)
        if quote_match:
            answer_candidate = quote_match.group(1).strip()
        else:
            # Take the first line or sentence as the answer
            lines = after_indicator.split('\n')
            if lines:
                answer_candidate = lines[0].strip()
        
        # Remove common suffixes that might be part of reasoning
        for suffix in ["。", ".", "，", ",", "；", ";"]:
            if answer_candidate.endswith(suffix):
                answer_candidate = answer_candidate[:-len(suffix)]
        
        # Remove asterisks and formatting
        answer_candidate = re.sub(r'\*+', '', answer_candidate)
        answer_candidate = answer_candidate.strip()
        
        # Check if we have a reasonable answer length for TCM syndrome names
        if answer_candidate and len(answer_candidate) > 0 and len(answer_candidate) <= 20:
            return answer_candidate
    
    # Strategy 3: Trunk response - keep only last few words for direct answers
    if answer_type == "direct":
        words = response.split()
        if len(words) > 10:  # If response is too long, keep last few words
            return " ".join(words[-5:])
    
    # Strategy 4: For multiple choice, look for single letter answers at the end
    if answer_type == "multiple_choice":
        # Look for isolated letters (A, B, C, D, E) at the end
        lines = response.split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line and len(line) <= 5:  # Short lines more likely to be answers
                # Check if it contains a single choice letter
                choice_match = re.search(r'\b([A-E])\b', line)
                if choice_match:
                    return choice_match.group(1)
    
    # Strategy 5: Return the last non-empty line if it's short enough
    lines = [line.strip() for line in response.split('\n') if line.strip()]
    if lines:
        last_line = lines[-1]
        if len(last_line) <= 50:  # Reasonable answer length
            return last_line
    
    # Fallback: return original response
    return original_response


def parse_options(options):
    """Parse options into formatted string with A, B, C, D, E format"""
    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    choices_str = "\n".join([f"{option_letter}. {option}" for option_letter, option in zip(option_letters, options)])
    return choices_str


def tcm_sd_doc_to_text_multiple_choice(doc, lmms_eval_specific_kwargs=None):
    """Convert document to text prompt for TCM multiple choice questions"""
    prompt = doc["prompt"]
    
    if lmms_eval_specific_kwargs is None:
        return prompt
    
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    
    return f"{pre_prompt}{prompt}{post_prompt}"


def tcm_sd_doc_to_text_direct(doc, lmms_eval_specific_kwargs=None):
    """Convert document to text prompt for TCM direct diagnosis"""
    prompt = doc["prompt"]
    
    if lmms_eval_specific_kwargs is None:
        return prompt
    
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    
    return f"{pre_prompt}{prompt}{post_prompt}"


def tcm_sd_doc_to_text_rc_five(doc, lmms_eval_specific_kwargs=None):
    """Convert document to text prompt for TCM reading comprehension (five options)"""
    prompt = doc["prompt"]
    
    if lmms_eval_specific_kwargs is None:
        return prompt
    
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    
    return f"{pre_prompt}{prompt}{post_prompt}"


def tcm_sd_doc_to_text_rc_all(doc, lmms_eval_specific_kwargs=None):
    """Convert document to text prompt for TCM reading comprehension (all options)"""
    prompt = doc["prompt"]
    
    if lmms_eval_specific_kwargs is None:
        return prompt
    
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    
    return f"{pre_prompt}{prompt}{post_prompt}"


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
    """Process results for TCM direct diagnosis with CoT filtering"""
    from lmms_eval.api.metrics import exact_match_hf_evaluate
    
    result = results[0]
    
    # Handle different response formats
    if hasattr(result, 'doc_ids'):
        # ResponseWithDocIds object
        raw_pred = str(result).strip()
        retrieved_doc_ids = result.doc_ids
    elif isinstance(result, dict):
        # Legacy dict format
        raw_pred = result["text"].strip()
        retrieved_doc_ids = result.get("doc_ids", [])
    else:
        # String format
        raw_pred = result.strip()
        retrieved_doc_ids = []
    
    # Apply CoT filtering to extract the final answer
    pred = filter_cot_response(raw_pred, answer_type="direct")
    
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
        "submission": {doc["user_id"]: pred},
        "cot_info": {
            "raw_response": raw_pred,
            "filtered_response": pred
        }
    }


def tcm_sd_process_results_multiple_choice(doc, results):
    """Process results for TCM multiple choice questions with CoT filtering"""
    from lmms_eval.api.metrics import exact_match_hf_evaluate
    
    result = results[0]
    
    # Handle different response formats
    if hasattr(result, 'doc_ids'):
        # ResponseWithDocIds object
        raw_pred = str(result).strip()
        retrieved_doc_ids = result.doc_ids
    elif isinstance(result, dict):
        # Legacy dict format
        raw_pred = result["text"].strip()
        retrieved_doc_ids = result.get("doc_ids", [])
    else:
        # String format
        raw_pred = result.strip()
        retrieved_doc_ids = []
    
    # Apply CoT filtering to extract the final answer
    filtered_pred = filter_cot_response(raw_pred, answer_type="multiple_choice")
        
    options = doc["options"]
    index2ans, all_choices = get_multi_choice_info(options)
    parsed_pred = parse_multi_choice_response(filtered_pred, all_choices, index2ans)
    
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
        "submission": {doc["user_id"]: parsed_pred},
        "cot_info": {
            "raw_response": raw_pred,
            "filtered_response": filtered_pred,
            "parsed_choice": parsed_pred
        }
    }


def tcm_sd_process_results_rc_five(doc, results):
    """Process results for TCM reading comprehension (five options) with CoT filtering"""
    from lmms_eval.api.metrics import exact_match_hf_evaluate
    
    result = results[0]
    
    # Handle different response formats
    if hasattr(result, 'doc_ids'):
        # ResponseWithDocIds object
        raw_pred = str(result).strip()
        retrieved_doc_ids = result.doc_ids
    elif isinstance(result, dict):
        # Legacy dict format
        raw_pred = result["text"].strip()
        retrieved_doc_ids = result.get("doc_ids", [])
    else:
        # String format
        raw_pred = result.strip()
        retrieved_doc_ids = []
    
    # Apply CoT filtering to extract the final answer
    pred = filter_cot_response(raw_pred, answer_type="direct")
    
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
        "submission": {doc["user_id"]: pred},
        "cot_info": {
            "raw_response": raw_pred,
            "filtered_response": pred
        }
    }


def tcm_sd_process_results_rc_all(doc, results):
    """Process results for TCM reading comprehension (all options) with CoT filtering"""
    from lmms_eval.api.metrics import exact_match_hf_evaluate
    
    result = results[0]
    
    # Handle different response formats
    if hasattr(result, 'doc_ids'):
        # ResponseWithDocIds object
        raw_pred = str(result).strip()
        retrieved_doc_ids = result.doc_ids
    elif isinstance(result, dict):
        # Legacy dict format
        raw_pred = result["text"].strip()
        retrieved_doc_ids = result.get("doc_ids", [])
    else:
        # String format
        raw_pred = result.strip()
        retrieved_doc_ids = []
    
    # Apply CoT filtering to extract the final answer
    pred = filter_cot_response(raw_pred, answer_type="direct")
    
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
        "submission": {doc["user_id"]: pred},
        "cot_info": {
            "raw_response": raw_pred,
            "filtered_response": pred
        }
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
