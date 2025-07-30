import json
import os
import re
from pathlib import Path
from typing import Dict, List


def load_json_data(filename: str) -> List[Dict]:
    """Load data from JSON file in the tcmeval_sdt_tasks directory"""
    data_dir = Path(__file__).parent / "tcmeval_sdt_tasks"
    file_path = data_dir / filename
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def tcmeval_sdt_task1_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Convert task1 document to text prompt for info extraction"""
    return doc["prompt"]


def tcmeval_sdt_task1_doc_to_target(doc):
    """Convert task1 document to target answer"""
    return doc["expected_answer"]


def tcmeval_sdt_task2_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Convert task2 document to text prompt for multiple choice"""
    return doc["prompt"]


def tcmeval_sdt_task2_doc_to_target(doc):
    """Convert task2 document to target answer"""
    return doc["expected_answer"]


def tcmeval_sdt_task3_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Convert task3 document to text prompt for multiple choice"""
    return doc["prompt"]


def tcmeval_sdt_task3_doc_to_target(doc):
    """Convert task3 document to target answer"""
    return doc["expected_answer"]


def tcmeval_sdt_task4_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Convert task4 document to text prompt for open-ended diagnosis"""
    return doc["prompt"]


def tcmeval_sdt_task4_doc_to_target(doc):
    """Convert task4 document to target answer"""
    return doc["expected_answer"]


def parse_multiple_choice_response(response: str, all_choices: List[str] = None) -> str:
    """
    Parse multiple choice response to extract letter choices.
    Returns semicolon-separated letters for multiple selections.
    """
    if not response:
        return ""
    
    # Clean response
    response = response.strip()
    
    # Pattern to match letters followed by semicolons or at the end
    pattern = r'\b([A-J])\b'
    matches = re.findall(pattern, response.upper())
    
    if matches:
        # Remove duplicates while preserving order
        unique_matches = []
        for match in matches:
            if match not in unique_matches:
                unique_matches.append(match)
        return ";".join(unique_matches)
    
    return ""


def tcmeval_sdt_task1_process_results(doc, results):
    """Process results for task1 (info extraction) using recall-based scoring"""
    pred = results[0] if results else ""
    target = doc["expected_answer"]
    
    # Parse expert annotations (A) and model predictions (B)
    # Split by semicolon and clean each item
    def parse_clinical_info(text):
        if not text or text.strip() == "":
            return set()
        items = [item.strip().lower() for item in text.split(';') if item.strip()]
        return set(items)
    
    expert_annotations = parse_clinical_info(target)  # A
    model_predictions = parse_clinical_info(pred)     # B
    
    # Calculate S_c = |A ∩ B| / |A|
    if len(expert_annotations) == 0:
        # If no expert annotations, score is 1.0 if model also predicts nothing
        score = 1.0 if len(model_predictions) == 0 else 0.0
    else:
        intersection = expert_annotations.intersection(model_predictions)
        score = len(intersection) / len(expert_annotations)
    
    return {
        "exact_match": score,
        "case_id": doc.get("case_id", "unknown"),
        "expert_count": len(expert_annotations),
        "model_count": len(model_predictions),
        "intersection_count": len(intersection) if len(expert_annotations) > 0 else 0
    }


def tcmeval_sdt_task2_process_results(doc, results):
    """Process results for task2 (pathogenesis reasoning) using S_p = |A ∩ B| / (|A| + |A ∩ B'|)"""
    pred = results[0] if results else ""
    target = doc["expected_answer"]
    
    # Parse the prediction to extract letter choices
    parsed_pred = parse_multiple_choice_response(pred)
    
    # Parse expert annotations (A) and model predictions (B)
    def parse_choices(text):
        if not text or text.strip() == "":
            return set()
        choices = [choice.strip().upper() for choice in text.split(';') if choice.strip()]
        return set(choices)
    
    expert_choices = parse_choices(target)  # A
    model_choices = parse_choices(parsed_pred)  # B
    
    # Calculate S_p = |A ∩ B| / (|A| + |A ∩ B'|)
    if len(expert_choices) == 0:
        # If no expert choices, score is 1.0 if model also predicts nothing
        score = 1.0 if len(model_choices) == 0 else 0.0
        wrong_predictions = len(model_choices)
    else:
        correct_predictions = len(expert_choices.intersection(model_choices))  # |A ∩ B|
        wrong_predictions = len(model_choices - expert_choices)  # |A ∩ B'| (model wrong choices)
        
        denominator = len(expert_choices) + wrong_predictions
        score = correct_predictions / denominator if denominator > 0 else 0.0
    
    return {
        "exact_match": score,
        "case_id": doc.get("case_id", "unknown"),
        "parsed_pred": parsed_pred,
        "target": target,
        "expert_count": len(expert_choices),
        "model_count": len(model_choices),
        "correct_predictions": len(expert_choices.intersection(model_choices)) if len(expert_choices) > 0 else 0,
        "wrong_predictions": wrong_predictions
    }


def tcmeval_sdt_task3_process_results(doc, results):
    """Process results for task3 (syndrome reasoning) using S_s = |A ∩ B| / (|A| + |A ∩ B'|)"""
    pred = results[0] if results else ""
    target = doc["expected_answer"]
    
    # Parse the prediction to extract letter choices
    parsed_pred = parse_multiple_choice_response(pred)
    
    # Parse expert annotations (A) and model predictions (B)
    def parse_choices(text):
        if not text or text.strip() == "":
            return set()
        choices = [choice.strip().upper() for choice in text.split(';') if choice.strip()]
        return set(choices)
    
    expert_choices = parse_choices(target)  # A
    model_choices = parse_choices(parsed_pred)  # B
    
    # Calculate S_s = |A ∩ B| / (|A| + |A ∩ B'|)
    if len(expert_choices) == 0:
        # If no expert choices, score is 1.0 if model also predicts nothing
        score = 1.0 if len(model_choices) == 0 else 0.0
        wrong_predictions = len(model_choices)
    else:
        correct_predictions = len(expert_choices.intersection(model_choices))  # |A ∩ B|
        wrong_predictions = len(model_choices - expert_choices)  # |A ∩ B'| (model wrong choices)
        
        denominator = len(expert_choices) + wrong_predictions
        score = correct_predictions / denominator if denominator > 0 else 0.0
    
    return {
        "exact_match": score,
        "case_id": doc.get("case_id", "unknown"),
        "parsed_pred": parsed_pred,
        "target": target,
        "expert_count": len(expert_choices),
        "model_count": len(model_choices),
        "correct_predictions": len(expert_choices.intersection(model_choices)) if len(expert_choices) > 0 else 0,
        "wrong_predictions": wrong_predictions
    }


def longest_common_subsequence(x, y):
    """Calculate the length of longest common subsequence between two sequences"""
    m, n = len(x), len(y)
    # Create DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i-1] == y[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]


def calculate_rouge_l(pred_text, ref_text):
    """Calculate ROUGE-L score between predicted and reference text"""
    # Tokenize texts (split by characters for Chinese text)
    pred_tokens = list(pred_text.strip())
    ref_tokens = list(ref_text.strip())
    
    if len(pred_tokens) == 0 and len(ref_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0
    
    # Calculate LCS length
    lcs_length = longest_common_subsequence(pred_tokens, ref_tokens)
    
    # Calculate precision and recall
    precision = lcs_length / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
    recall = lcs_length / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
    
    # Calculate F1-score (ROUGE-L with β=1)
    if precision + recall == 0:
        return 0.0
    
    rouge_l = 2 * precision * recall / (precision + recall)
    return rouge_l


def tcmeval_sdt_task4_process_results(doc, results):
    """Process results for task4 (explanatory summary) using ROUGE-L scoring"""
    pred = results[0] if results else ""
    target = doc["expected_answer"]
    
    # Clean texts - remove common prefixes
    pred_clean = pred.strip()
    target_clean = target.strip()
    
    # Remove common prefixes like "辨证：" if present
    if pred_clean.startswith("辨证："):
        pred_clean = pred_clean[3:].strip()
    if target_clean.startswith("辨证："):
        target_clean = target_clean[3:].strip()
    
    # Calculate ROUGE-L score
    rouge_l_score = calculate_rouge_l(pred_clean, target_clean)
    
    return {
        "rouge_l": rouge_l_score,
        "case_id": doc.get("case_id", "unknown"),
        "pred_length": len(pred_clean),
        "target_length": len(target_clean),
        "lcs_length": longest_common_subsequence(list(pred_clean), list(target_clean))
    }

# Multiple choice regex filter for tasks 2 and 3
class MultiChoiceRegexFilter:
    """Filter for multiple choice responses to extract letter choices"""
    
    def __init__(self, **kwargs):
        pass
    
    def apply(self, resps, docs):
        """Apply filtering to extract multiple choice answers"""
        filtered_resps = []
        
        for resp_list in resps:
            if resp_list:
                # Parse the first response
                parsed = parse_multiple_choice_response(resp_list[0])
                filtered_resps.append(parsed if parsed else "")
            else:
                filtered_resps.append("")
        
        return filtered_resps


def tcmeval_sdt_weighted_aggregate(results_dict):
    """
    Calculate weighted final score for tcmeval_sdt tasks
    S_f = ω₁S_c + ω₂S_p + ω₃S_s + ω₄S_r
    
    Weights:
    - Task 1 (Clinical Info): ω₁ = 0.2 (20%)
    - Task 2 (Pathogenesis): ω₂ = 0.3 (30%) 
    - Task 3 (Syndrome): ω₃ = 0.4 (40%)
    - Task 4 (Summary): ω₄ = 0.1 (10%)
    """
    weights = {
        'tcmeval_sdt_task1': 0.2,  # S_c - Clinical information extraction
        'tcmeval_sdt_task2': 0.3,  # S_p - Pathogenesis reasoning
        'tcmeval_sdt_task3': 0.4,  # S_s - Syndrome reasoning
        'tcmeval_sdt_task4': 0.1,  # S_r - Explanatory summary
    }
    
    weighted_score = 0.0
    total_weight = 0.0
    task_scores = {}
    
    for task_name, weight in weights.items():
        if task_name in results_dict:
            task_result = results_dict[task_name]
            # Extract the appropriate score based on task type
            if isinstance(task_result, dict):
                if task_name == 'tcmeval_sdt_task4':
                    # Task 4 uses rouge_l metric
                    task_score = task_result.get('rouge_l', 0.0)
                else:
                    # Tasks 1-3 use exact_match metric
                    task_score = task_result.get('exact_match', 0.0)
            elif isinstance(task_result, (int, float)):
                task_score = task_result
            else:
                task_score = 0.0
                
            if isinstance(task_score, (int, float)):
                weighted_score += weight * task_score
                total_weight += weight
                task_scores[task_name] = task_score
    
    # Normalize by actual total weight (in case some tasks are missing)
    final_score = weighted_score / total_weight if total_weight > 0 else 0.0
    
    return {
        'tcmeval_sdt_final_score': final_score,
        'task_scores': task_scores,
        'weights_used': {k: v for k, v in weights.items() if k in results_dict},
        'total_weight': total_weight
    }
