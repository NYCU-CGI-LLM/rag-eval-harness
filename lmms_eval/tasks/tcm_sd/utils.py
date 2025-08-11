import json
import random
import re
from collections import defaultdict

import numpy as np
from loguru import logger as eval_logger


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
    
    # Strategy 1: Enhanced quote extraction - look for answers in quotes first
    quote_patterns = [
        # Various quote styles with TCM syndrome names - include all Unicode quote types
        r'[""\u201c\u201d''""]([^""\u201c\u201d''""\n]{2,15}证)[""\u201c\u201d''""]',  # TCM syndromes in quotes
        r'[""\u201c\u201d''""]([^""\u201c\u201d''""\n]{2,20})[""\u201c\u201d''""](?:[。．，,\s]*$)',  # General quoted answers at end
        r'最符合的.*?[：:：]?\s*[""\u201c\u201d''""]([^""\u201c\u201d''""\n]{2,15}证)[""\u201c\u201d''""]',  # TCM specific with quotes
        # Also try simple patterns for the specific case
        r'"([^"\n]{2,15}证)"',  # Simple double quotes
        r'"([^"\n]{2,15}证)"',  # Curly double quotes  
    ]
    
    for pattern in quote_patterns:
        matches = re.findall(pattern, response, re.IGNORECASE | re.MULTILINE)
        if matches:
            # Take the last match (most likely to be the final answer)
            extracted = matches[-1].strip()
            if extracted and len(extracted) <= 20 and '证' in extracted:
                return extracted
    
    # Strategy 2: Look for structured answer patterns
    # Handle "Answer:" patterns and TCM-specific patterns
    answer_patterns = [
        # Basic answer patterns - more specific
        r'答案[：:]\s*([^。\n]{2,15}证)(?:[。．\n]|$)',
        r'诊断[：:]\s*([^。\n]{2,15}证)(?:[。．\n]|$)', 
        r'证候[：:]\s*([^。\n]{2,15}证)(?:[。．\n]|$)',
        r'结论[：:]\s*([^。\n]{2,15}证)(?:[。．\n]|$)',
        
        # TCM-specific patterns - more precise
        r'证候名称[为]?[：:]?\s*([^。\n]{2,15}证)(?:[。．\n]|$)',
        r'最符合的.*?证候.*?[为是][：:]?\s*([^。\n]{2,15}证)',
        r'最符合的.*?证候[：:]?\s*([^。\n]{2,15}证)',
        r'证候.*?为[：:]?\s*([^。\n]{2,15}证)',
        
        # Bold/asterisk patterns
        r'\*\*([^*\n]{2,15}证)\*\*',
        
        # Final conclusion patterns
        r'(?:因此|所以|综上|最终|最后).*?[：:]?\s*([^。\n]{2,15}证)(?:[。．]|$)',
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
            extracted = re.sub(r'[\*\*]+', '', extracted)
            if extracted and not extracted.isspace() and len(extracted) <= 20 and '证' in extracted:
                return extracted
    
    # Strategy 3: Enhanced indicator-based extraction
    # Find the last occurrence of key indicators, but be more selective
    best_candidate = ""
    best_score = 0
    
    # Look for key indicators in reverse order (last occurrence first)
    key_indicators = ["最符合的", "证候为", "证候是", "诊断为", "诊断是", "结论", "综上", "因此", "最终"]
    
    for indicator in key_indicators:
        pos = response.lower().rfind(indicator.lower())
        if pos != -1:
            # Extract content after the indicator
            after_indicator = response[pos + len(indicator):].strip()
            
            # Clean up common prefixes and separators
            for sep in ["：", ":", "是", "为", "："]:
                if after_indicator.startswith(sep):
                    after_indicator = after_indicator[len(sep):].strip()
                    break
            
            # Look for quoted content first - handle all Unicode quote types
            quote_match = re.search(r'^[""\u201c\u201d''""]([^""\u201c\u201d''""\n]{2,15}证)[""\u201c\u201d''""]', after_indicator)
            if quote_match:
                candidate = quote_match.group(1).strip()
                if len(candidate) <= 20 and '证' in candidate:
                    return candidate
            
            # Look for TCM syndrome names in the first part
            syndrome_match = re.search(r'^([^。\n]{2,15}证)', after_indicator)
            if syndrome_match:
                candidate = syndrome_match.group(1).strip()
                # Clean up
                candidate = re.sub(r'^[：:：，,。．\s]+', '', candidate)
                candidate = re.sub(r'[。．，,\s]+$', '', candidate)
                if len(candidate) <= 20 and '证' in candidate:
                    # Score based on position and indicator quality
                    score = pos + (100 if indicator in ["最符合的", "证候为", "诊断为"] else 50)
                    if score > best_score:
                        best_candidate = candidate
                        best_score = score
    
    if best_candidate:
        return best_candidate
    
    # Strategy 4: Look for TCM syndromes in the last part of response
    if answer_type == "direct":
        # Look for any TCM syndrome names in the last few sentences
        sentences = re.split(r'[。．！!？?]', response)
        for sentence in reversed(sentences):
            sentence = sentence.strip()
            if sentence:
                # Find syndrome names ending with 证
                syndrome_match = re.search(r'([^。\n]{2,15}证)', sentence)
                if syndrome_match:
                    candidate = syndrome_match.group(1).strip()
                    if len(candidate) <= 20:
                        return candidate
        
        # If still no match, try last few words
        words = response.split()
        if len(words) > 10:
            # Look for syndrome in last 10 words
            last_words = " ".join(words[-10:])
            syndrome_match = re.search(r'([^。\n]{2,15}证)', last_words)
            if syndrome_match:
                return syndrome_match.group(1).strip()
    
    # Strategy 5: For multiple choice, look for single letter answers at the end
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
    
    # Strategy 6: Return the last non-empty line if it's short and contains 证
    lines = [line.strip() for line in response.split('\n') if line.strip()]
    if lines:
        for line in reversed(lines):
            if len(line) <= 20 and '证' in line:
                # Clean up the line
                line = re.sub(r'^[：:：，,。．\s]+', '', line)
                line = re.sub(r'[。．，,\s]+$', '', line)
                if line:
                    return line
    
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
