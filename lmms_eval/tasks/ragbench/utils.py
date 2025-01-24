import json
import os
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import requests
import yaml
from loguru import logger as eval_logger

NUM_SECONDS_TO_SLEEP = 5

RAG_METRICS = [
    "gpt_eval_rag_factual",
    "gpt_eval_rag_reasoning",
    "gpt_eval_rag_comprehensive",
    "gpt_eval_rag_diversity",
    "gpt_eval_rag_empowerment"
]

# 載入評估規則和角色定義
rule_dict = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rule.json"), "r"))
role_dict = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "role.json"), "r"))

# 載入配置文件
with open(Path(__file__).parent / "ragbench.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)
    config = yaml.safe_load("".join(safe_data))

GPT_EVAL_MODEL_NAME = config["metadata"]["gpt_eval_model_name"]

API_TYPE = os.getenv("API_TYPE", "openai")

if API_TYPE == "openai":
    API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
    API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
elif API_TYPE == "azure":
    API_URL = os.getenv("AZURE_ENDPOINT", "https://api.cognitive.microsoft.com/sts/v1.0/issueToken")
    API_KEY = os.getenv("AZURE_API_KEY", "YOUR_API_KEY")
    headers = {
        "api-key": API_KEY,
        "Content-Type": "application/json",
    }

def get_eval(content: str, max_tokens: int, retries: int = 5):
    """調用 API 進行評估"""
    global headers

    messages = [
        {
            "role": "system",
            "content": role_dict["role"],
        },
        {"role": "user", "content": content},
    ]

    payload = {
        "model": GPT_EVAL_MODEL_NAME,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }

    if API_TYPE == "azure":
        payload.pop("model")

    for attempt in range(retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            response_data = response.json()

            content = response_data["choices"][0]["message"]["content"].strip()
            if content != "":
                return content, response_data["model"]
            break

        except Exception as e:
            eval_logger.info(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt < retries:
                time.sleep(NUM_SECONDS_TO_SLEEP)
            else:
                eval_logger.error(f"All {retries} attempts failed. Last error message: {e}")
                return "", ""
    return "", ""

def parse_score(review):
    """解析評分結果"""
    try:
        score_pair = review.split("\n")[0]
        score_pair = score_pair.replace(",", " ")
        sp = score_pair.split(" ")
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            eval_logger.debug(f"Can not split: {review}. Returning [-1, -1]")
            return [-1, -1]
    except Exception as e:
        eval_logger.debug(f"Error: {e}. Returning [-1, -1]")
        return [-1, -1]

def ragbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """將文檔轉換為文本格式"""
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    return f"{pre_prompt}{doc['question']}{post_prompt}"

def ragbench_process_results(doc, result):
    """處理評估結果"""
    try:
        question = doc.get("question", "")
        context = doc.get("context", "")
        ans = doc.get("reference_answer", "")  # 資料集答案 (option)
        category = doc.get("category", "default")
        rule = rule_dict.get(category, rule_dict["default"])
        prompt = rule.get("prompt", "")

        # 隨機決定答案順序
        if random.random() < 0.5:
            answer1, answer2 = reference_answer, system_answer
            answer_order = [0, 1]  # 0 表示參考答案, 1 表示系統答案
        else:
            answer1, answer2 = system_answer, reference_answer
            answer_order = [1, 0]

        content = (f"[Question]\n{question}\n\n"
                  f"[Answer 1]\n{answer1}\n\n[End of Answer 1]\n\n"
                  f"[Answer 2]\n{answer2}\n\n[End of Answer 2]\n\n"
                  f"[Evaluation Instructions]\n{prompt}\n\n")

        review, model_name = get_eval(content, 1024)
        raw_scores = parse_score(review)

        # 根據答案順序重新排列分數
        scores = [raw_scores[answer_order.index(0)], raw_scores[answer_order.index(1)]]

    except Exception as e:
        eval_logger.error(f"Error for Question ID: {doc.get('question_id', 'Unknown')}: {e}")
        review = "Failed to Get a Proper Review."
        model_name = "Failed Request"
        scores = [-1, -1]

    metric = f"gpt_eval_rag_{category}"
    category_review_dict = {
        "question": question,
        "reference_answer": ans1,
        "system_answer": ans2,
        "category": category,
        "review": review,
        "scores": scores,
        "eval_model": model_name,
        "content": content
    }

    non_category_review_dict = deepcopy(category_review_dict)
    non_category_review_dict["scores"] = [-999, -999]

    data_dict = {}
    for m in RAG_METRICS:
        if m == metric:
            data_dict[m] = category_review_dict
        else:
            data_dict[m] = non_category_review_dict
    data_dict["gpt_eval_rag_all"] = category_review_dict

    return data_dict

def ragbench_aggregation(results, category):
    """聚合評估結果"""
    try:
        scores = []
        for result in results:
            if -999 in result["scores"]:
                continue
            scores.append(result["scores"])

        stats = np.asarray(scores).mean(0).tolist()
        stats = [round(x, 3) for x in stats]
        return round(stats[1] / stats[0] * 100, 1)  # 系統答案得分/參考答案得分 的百分比
    except Exception as e:
        eval_logger.info(f"Error in ragbench_aggregation: {e}, and in category: {category}")
        return None

# 為每個評估維度創建聚合函數
def ragbench_factual_aggregation(results):
    return ragbench_aggregation(results, "factual")

def ragbench_reasoning_aggregation(results):
    return ragbench_aggregation(results, "reasoning")

def ragbench_comprehensive_aggregation(results):
    return ragbench_aggregation(results, "comprehensive")

def ragbench_diversity_aggregation(results):
    return ragbench_aggregation(results, "diversity")

def ragbench_empowerment_aggregation(results):
    return ragbench_aggregation(results, "empowerment")

def ragbench_all_aggregation(results):
    return ragbench_aggregation(results, "all")