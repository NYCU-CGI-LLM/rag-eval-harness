import copy
import json
import logging
from typing import Any, Dict, List, Literal, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from packaging.version import Version

from lmms_eval.loggers.utils import _handle_non_serializable, remove_none_pattern


def get_wandb_printer() -> Literal["Printer"]:
    """Returns a wandb printer instance for pretty stdout."""
    from wandb.sdk.lib.printer import new_printer
    printer = new_printer()
    return printer


class WandbLogger:
    def __init__(self, **kwargs) -> None:
        """Attaches to wandb logger if already initialized. Otherwise, passes kwargs to wandb.init()

        Args:
            kwargs Optional[Any]: Arguments for configuration.

        Parse and log the results returned from evaluator.simple_evaluate() with:
            wandb_logger.post_init(results)
            wandb_logger.log_eval_result()
            wandb_logger.log_eval_samples(results["samples"])
        """
        try:
            import wandb

            assert Version(wandb.__version__) >= Version("0.13.6")
            if Version(wandb.__version__) < Version("0.13.6"):
                wandb.require("report-editing:v0")
        except Exception as e:
            logger.warning("To use the wandb reporting functionality please install wandb>=0.13.6.\n" "To install the latest version of wandb run `pip install wandb --upgrade`\n" f"{e}")

        self.wandb_args: Dict[str, Any] = kwargs
        self.weave_initialized = False

        # initialize a W&B run
        if wandb.run is None:
            self.run = wandb.init(**self.wandb_args)
            
            # Initialize weave and return True/False for success
            try:
                import weave
                weave_args = {"project_name": self.wandb_args["project"]} if "project" in self.wandb_args else {}
                weave.init(**weave_args)
                self.weave_initialized = True
                logger.info("Weave initialized successfully")
            except Exception as e:
                self.weave_initialized = False
                logger.warning(f"Failed to initialize weave: {e}")
                
        else:
            self.run = wandb.run

        self.printer = get_wandb_printer()

    def post_init(self, results: Dict[str, Any]) -> None:
        self.results: Dict[str, Any] = copy.deepcopy(results)
        self.task_names: List[str] = list(results.get("results", {}).keys())
        self.group_names: List[str] = list(results.get("groups", {}).keys())

    def _get_config(self) -> Dict[str, Any]:
        """Get configuration parameters."""
        self.task_configs = self.results.get("configs", {})
        cli_configs = self.results.get("config", {})
        configs = {
            "task_configs": self.task_configs,
            "cli_configs": cli_configs,
        }

        return configs

    def _prepare_results_for_wandb(self) -> Dict[str, Any]:
        """Prepare complete results for wandb logging, filtering non-numeric values."""
        results = copy.deepcopy(self.results.get("results", {}))
        
        wandb_data = {}
        
        # Log results with task prefixes for clarity, but filter out problematic values
        for task_name, task_results in results.items():
            for metric_name, metric_value in task_results.items():
                # Skip non-metric fields and problematic values
                if (metric_name == "submission" or 
                    metric_name == "alias" or
                    isinstance(metric_value, (list, dict))):
                    continue
                
                # Only include numeric values or convertible strings
                if isinstance(metric_value, (int, float)):
                    clean_metric_name, _ = remove_none_pattern(metric_name)
                    key = f"{task_name}/{clean_metric_name}"
                    wandb_data[key] = metric_value
                elif isinstance(metric_value, str):
                    try:
                        # Try to convert string to number
                        numeric_value = float(metric_value)
                        clean_metric_name, _ = remove_none_pattern(metric_name)
                        key = f"{task_name}/{clean_metric_name}"
                        wandb_data[key] = numeric_value
                    except (ValueError, TypeError):
                        # Skip non-numeric strings
                        continue
        
        return wandb_data

    def _log_complete_results_table(self, samples: Dict[str, List[Dict[str, Any]]] = None) -> None:
        """Generate and log complete evaluation results as comprehensive tables to W&B."""
        import wandb
        
        # Create simplified table focused on metrics only
        columns = [
            "Task",
            "Version",
            "Filter",
            "num_fewshot",
            "Metric",
            "Value",
            "Stderr"
        ]
        
        table = wandb.Table(columns=columns)
        results = copy.deepcopy(self.results)
        
        # Log complete results for better traceability
        for task_name, task_results in results.get("results", {}).items():
            if task_name in self.group_names:
                continue
                
            version = results.get("versions", {}).get(task_name, "N/A")
            n_shot = results.get("n-shot", {}).get(task_name, "N/A")
            
            for metric_full, value in task_results.items():
                metric, _, filter_name = metric_full.partition(",")
                
                # Skip non-metric fields
                if (metric.endswith("_stderr") or 
                    metric == "alias" or 
                    metric == "submission" or
                    isinstance(value, list) or
                    isinstance(value, dict)):
                    continue
                
                # Only process numeric metrics
                if not isinstance(value, (int, float)):
                    try:
                        float(value)
                    except (ValueError, TypeError):
                        continue
                
                # Get stderr if available
                stderr_key = f"{metric}_stderr,{filter_name}" if filter_name else f"{metric}_stderr"
                stderr = task_results.get(stderr_key, "")
                if stderr and stderr != "N/A" and isinstance(stderr, (int, float)):
                    stderr = f"{stderr:.4f}"
                else:
                    stderr = "N/A"
                
                table.add_data(
                    task_name,
                    version,
                    filter_name or "none",
                    str(n_shot),
                    metric,
                    f"{value:.4f}" if isinstance(value, float) else str(value),
                    stderr
                )
        
        self.run.log({"evaluation/results_summary": table})
        
        # Also log groups if available
        if "groups" in results:
            group_table = wandb.Table(columns=columns)
            for group_name, group_results in results.get("groups", {}).items():
                version = results.get("versions", {}).get(group_name, "N/A")
                n_shot = results.get("n-shot", {}).get(group_name, "N/A")
                
                for metric_full, value in group_results.items():
                    metric, _, filter_name = metric_full.partition(",")
                    
                    # Skip non-metric fields
                    if (metric.endswith("_stderr") or 
                        metric == "alias" or 
                        metric == "submission" or
                        isinstance(value, list) or
                        isinstance(value, dict)):
                        continue

                    # Only process numeric metrics
                    if not isinstance(value, (int, float)):
                        try:
                            float(value)
                        except (ValueError, TypeError):
                            continue
                    
                    stderr_key = f"{metric}_stderr,{filter_name}" if filter_name else f"{metric}_stderr"
                    stderr = group_results.get(stderr_key, "")
                    if stderr and stderr != "N/A" and isinstance(stderr, (int, float)):
                        stderr = f"{stderr:.4f}"
                    else:
                        stderr = "N/A"
                    
                    group_table.add_data(
                        group_name,
                        version,
                        filter_name or "none", 
                        str(n_shot),
                        metric,
                        f"{value:.4f}" if isinstance(value, float) else str(value),
                        stderr
                    )
            
            self.run.log({"evaluation/group_summary": group_table})

    def _log_results_as_artifact(self) -> None:
        """Log results as JSON artifact to W&B."""
        import wandb

        dumped = json.dumps(self.results, indent=2, default=_handle_non_serializable, ensure_ascii=False)
        artifact = wandb.Artifact("results", type="eval_results")
        with artifact.new_file("results.json", mode="w", encoding="utf-8") as f:
            f.write(dumped)
        self.run.log_artifact(artifact)

    def log_eval_result(self, samples: Dict[str, List[Dict[str, Any]]] = None) -> None:
        """Log evaluation results to W&B with complete data for better traceability."""
        # Log configs to wandb
        configs = self._get_config()
        self.run.config.update(configs)

        # Prepare and log complete results without data loss
        wandb_data = self._prepare_results_for_wandb()
        self.run.log(wandb_data)
        
        # Log comprehensive results table for problem tracing and answer analysis
        self._log_complete_results_table()
        
        # Log Q&A samples table for better traceability (always log this)
        if samples:
            self._log_samples_table(samples)
        
        # Log the complete results dict as json to W&B Artifacts for full traceability
        self._log_results_as_artifact()
        
        # Log weave initialization status
        self.run.summary.update({"weave_initialized": self.weave_initialized})

    def _generate_dataset(self, data: List[Dict[str, Any]], config: Dict[str, Any]) -> pd.DataFrame:
        """Generate a simplified dataset from evaluation data keeping all important information.

        Args:
            data (List[Dict[str, Any]]): The data to generate a dataset for.
            config (Dict[str, Any]): The configuration of the task.

        Returns:
            pd.DataFrame: A dataframe that is ready to be uploaded to W&B with complete information.
        """
        # Extract basic information
        ids = [x.get("doc_id", i) for i, x in enumerate(data)]
        targets = [str(x.get("target", "")) for x in data]
        
        # Extract questions and context
        questions = []
        context = []
        generated_answers = []
        expected_answers = []
        
        for x in data:
            # Try to extract question from various possible locations
            question = x.get("question", "")
            if not question and "arguments" in x and x["arguments"]:
                question = str(x["arguments"][0][0]) if len(x["arguments"][0]) > 0 else ""
            questions.append(question[:500])  # Truncate for readability
            
            # Extract context
            ctx = x.get("context", x.get("doc", ""))
            context.append(str(ctx)[:500])  # Truncate for readability
            
            # Extract generated answers
            generated_answer = x.get("generated_answer", "")
            if not generated_answer and "resps" in x and x["resps"]:
                if isinstance(x["resps"][0], list) and len(x["resps"][0]) > 0:
                    generated_answer = str(x["resps"][0][0])
                else:
                    generated_answer = str(x["resps"][0])
            generated_answers.append(generated_answer[:500])
            
            # Extract expected answers
            expected_answer = x.get("expected_answer", x.get("target", ""))
            expected_answers.append(str(expected_answer)[:500])
        
        # Create comprehensive dataframe
        df_data = {
            "doc_id": ids,
            "question": questions,
            "context": context,
            "generated_answer": generated_answers,
            "expected_answer": expected_answers,
            "target": targets,
            "output_type": config.get("output_type", "unknown")
        }
        
        # Add all metrics without complex processing
        metrics_list = config.get("metric_list", [])
        for metric_config in metrics_list:
            if isinstance(metric_config, dict):
                metric_name = metric_config.get("metric", "unknown_metric")
            else:
                metric_name = str(metric_config)
            
            # Extract metric values directly
            metric_values = []
            for x in data:
                if metric_name in x:
                    value = x[metric_name]
                    # Handle complex metric values
                    if isinstance(value, (list, tuple)) and len(value) > 0:
                        metric_values.append(value[0] if isinstance(value[0], (int, float)) else str(value[0]))
                    else:
                        metric_values.append(value)
                else:
                    metric_values.append(None)
            
            df_data[metric_name] = metric_values
        
        # Add raw data for debugging
        df_data["raw_arguments"] = [str(x.get("arguments", ""))[:200] for x in data]
        df_data["raw_resps"] = [str(x.get("resps", ""))[:200] for x in data]
        df_data["filtered_resps"] = [str(x.get("filtered_resps", ""))[:200] for x in data]

        return pd.DataFrame(df_data)

    def _log_samples_as_artifact(self, data: List[Dict[str, Any]], task_name: str) -> None:
        import wandb

        # log the samples as an artifact
        dumped = json.dumps(
            data,
            indent=2,
            default=_handle_non_serializable,
            ensure_ascii=False,
        )
        artifact = wandb.Artifact(f"{task_name}", type="samples_by_task")
        with artifact.new_file(f"{task_name}_eval_samples.json", mode="w", encoding="utf-8") as f:
            f.write(dumped)
        self.run.log_artifact(artifact)
        # artifact.wait()

    def _log_samples_table(self, samples: Dict[str, List[Dict[str, Any]]]) -> None:
        """Log a dedicated table with questions, generated answers, and expected answers."""
        import wandb
        
        if not samples:
            logger.warning("Samples is None or empty!")
            return
        
        # Create table specifically for Q&A traceability with correctness
        qa_columns = [
            "Task",
            "Doc_ID", 
            "Question",
            "Generated_Answer",
            "Expected_Answer",
            "Is_Correct",
            "Context"
        ]
        
        qa_table = wandb.Table(columns=qa_columns)
        total_rows_added = 0
        
        for task_name, task_samples in samples.items():
            # Check if task should be skipped
            if task_name in getattr(self, 'group_names', []):
                continue
                
            if not task_samples:
                continue
                
            # Process all samples
            for i, sample in enumerate(task_samples):
                # Extract question
                question = sample.get("question", "")
                if not question and "arguments" in sample and sample["arguments"]:
                    try:
                        if isinstance(sample["arguments"], list) and len(sample["arguments"]) > 0:
                            if isinstance(sample["arguments"][0], list) and len(sample["arguments"][0]) > 0:
                                question = str(sample["arguments"][0][0])
                            else:
                                question = str(sample["arguments"][0])
                    except (IndexError, TypeError) as e:
                        logger.warning(f"Error extracting question: {e}")
                        question = ""
                
                # Extract generated answer
                generated_answer = sample.get("generated_answer", "")
                if not generated_answer and "resps" in sample and sample["resps"]:
                    try:
                        if isinstance(sample["resps"], list) and len(sample["resps"]) > 0:
                            if isinstance(sample["resps"][0], list) and len(sample["resps"][0]) > 0:
                                generated_answer = str(sample["resps"][0][0])
                            else:
                                generated_answer = str(sample["resps"][0])
                    except (IndexError, TypeError) as e:
                        logger.warning(f"Error extracting generated_answer: {e}")
                        generated_answer = ""
                
                # Extract expected answer
                expected_answer = sample.get("expected_answer", sample.get("target", ""))
                
                # Extract context
                context = sample.get("context", sample.get("doc", ""))
                if isinstance(context, dict):
                    context = str(context.get("text", context))
                
                # Check correctness - compare generated and expected answers
                is_correct = False
                if generated_answer and expected_answer:
                    # Simple string comparison (case-insensitive, stripped)
                    gen_clean = str(generated_answer).strip().lower()
                    exp_clean = str(expected_answer).strip().lower()
                    is_correct = gen_clean == exp_clean
                    
                    # For multiple choice, also check if the answer is contained
                    if not is_correct and len(gen_clean) == 1 and len(exp_clean) == 1:
                        is_correct = gen_clean == exp_clean
                    elif not is_correct:
                        # Check if expected answer is contained in generated answer
                        is_correct = exp_clean in gen_clean or gen_clean in exp_clean
                
                # Also check exact_match from sample if available
                if "exact_match" in sample:
                    is_correct = bool(sample["exact_match"])
                
                # Ensure all data is valid strings and not None - no truncation
                raw_doc_id = sample.get("doc_id")

                doc_id = raw_doc_id if raw_doc_id is not None and str(raw_doc_id).strip() else f"{task_name}_{i}"
                question_str = str(question) if question else "No question"
                generated_str = str(generated_answer) if generated_answer else "No answer"
                expected_str = str(expected_answer) if expected_answer else "No target"
                context_str = str(context) if context else "No context"
                
                # Only log doc_id issues for debugging
                if raw_doc_id is None or not str(raw_doc_id).strip():
                    logger.warning(f"Sample {i} has empty doc_id: {raw_doc_id}, using fallback: {doc_id}")
                
                # Add to table with full content (no truncation)
                qa_table.add_data(
                    str(task_name),
                    str(doc_id),
                    question_str,
                    generated_str,
                    expected_str,
                    bool(is_correct),
                    context_str
                )
                total_rows_added += 1
        
        # Always log this table, even if wandb_log_samples is False
        self.run.log({"evaluation/qa_samples": qa_table})

    def log_eval_samples(self, samples: Dict[str, List[Dict[str, Any]]]) -> None:
        """Log evaluation samples to W&B with complete sample data for better traceability.

        Args:
            samples (Dict[str, List[Dict[str, Any]]]): Evaluation samples for each task.
        """
        # Log Q&A table for traceability
        self._log_samples_table(samples)
        
        # Log all tasks directly without complex grouping logic
        for task_name, task_samples in samples.items():
            if task_name in self.group_names:
                continue
                
            if not task_samples:
                continue
                
            task_config = self.task_configs.get(task_name, {})
            
            # Generate comprehensive dataset with all sample information
            df = self._generate_dataset(task_samples, task_config)
            df["task_name"] = task_name
            
            # Log the complete sample data as W&B Table for easy analysis
            self.run.log({f"{task_name}_detailed_samples": df})

            # Log the raw samples as JSON artifact for full traceability
            self._log_samples_as_artifact(task_samples, task_name)
        
        # Handle groups separately if they exist
        if hasattr(self, 'group_names') and self.group_names:
            for group_name in self.group_names:
                if group_name in samples:
                    group_samples = samples[group_name]
                    if group_samples:
                        group_config = self.task_configs.get(group_name, {})
                        df = self._generate_dataset(group_samples, group_config)
                        df["group_name"] = group_name
                        self.run.log({f"{group_name}_group_samples": df})
                        self._log_samples_as_artifact(group_samples, f"group_{group_name}")
                        
        # Log summary of weave status and samples count
        sample_summary = {
            "total_tasks": len(samples),
            "total_samples": sum(len(task_samples) for task_samples in samples.values()),
            "weave_status": "initialized" if self.weave_initialized else "failed"
        }
        self.run.summary.update(sample_summary)
