import sympy
import re
from evals.benchmarks.utils import *

def answer_extraction_math(text: str) -> str:
    """
    Extracts the final answer from the generated text for the MATH dataset.
    Supports formats like '\\boxed{...}' and extracts expressions between the first and last '$' signs.
    """
    if "\\boxed" in text:
        return remove_boxed(last_boxed_only_string(text))
    elif "Answer:" in text:
        return text.split("Answer:")[-1]
    elif "###" in text:
        return text.split("###")[-1]
    else:
        return text



def answer_extraction_gsm8k(generated_text: str) -> str:
    """
    Extracts the final answer from the generated text for the GSM8k dataset.
    """
    # Try to find patterns like 'Answer: 42' or '#### 42'
    match = re.findall(r'####\s*(.*)', generated_text)
    if match:
        return match[-1].strip()
    match = re.findall(r'Answer:\s*(.*)', generated_text)
    if match:
        return match[-1].strip()
    # If no pattern matched, return the last numerical value
    numbers = re.findall(r'[-+]?\d*\.\d+|\d+', generated_text)
    if numbers:
        return numbers[-1].strip()
    # Fallback: Return the entire text
    return generated_text.strip()

def answer_extraction_aqua_rat(generated_text: str) -> str:
    """
    Extracts the selected option from the generated text for the AQUA-RAT dataset.
    """
    # Look for 'Answer: A', 'Answer: B', etc.
    match = re.findall(r'Answer:\s*([A-E])', generated_text, re.IGNORECASE)
    if match:
        return match[-1].upper()
    # Fallback: Return the last capital letter between A and E
    match = re.findall(r'\b([A-E])\b', generated_text.upper())
    if match:
        return match[-1]
    # Fallback: Return the entire text
    return generated_text.strip()
