import re
from lmms_eval.filters.extraction import ExtendedRegexFilter

def doc_to_text(doc) -> str:
    option_choices = {
        "A": doc["ending0"],
        "B": doc["ending1"],
        "C": doc["ending2"],
        "D": doc["ending3"],
    }
    answers = "".join((f"{k}. {v}\n") for k, v in option_choices.items())
    return f"Question: {doc['sent1']}\n{answers}Answer with the option letter (A, B, C, or D):"

def doc_to_target(doc) -> int:
    """Return the integer label for generation tasks - framework will convert to choice letter"""
    return doc["label"]

class MultiChoiceRegexFilter(ExtendedRegexFilter):
    """Filter to extract multiple choice answers from generated text."""
    
    def __init__(self, regex_pattern: str = r"([A-D])", group_select: int = 0, 
                 ignore_case: bool = True, ignore_punctuation: bool = True, **kwargs):
        super().__init__(
            regex_pattern=regex_pattern,
            group_select=group_select,
            ignore_case=ignore_case,
            ignore_punctuation=ignore_punctuation,
            fallback="A",
            **kwargs
        )
    
    def apply(self, resps, docs):
        """Extract the first matching choice letter from each response."""
        filtered_resps = []
        
        for r, doc in zip(resps, docs):
            filtered = []
            for resp in r:
                # Clean up the response text
                cleaned_resp = resp.upper().strip()
                
                # Try to match A, B, C, or D in the response
                match = re.search(r'\b([A-D])\b', cleaned_resp)
                if match:
                    filtered.append(match.group(1))
                else:
                    # Fallback: try to find any single letter that could be an answer
                    letter_match = re.search(r'([A-D])', cleaned_resp)
                    if letter_match:
                        filtered.append(letter_match.group(1))
                    else:
                        # Final fallback
                        filtered.append("A")
            
            # Return the first (and likely only) filtered response
            filtered_resps.append(filtered[0] if filtered else "A")
        
        return filtered_resps
    