import json
from typing import List, Dict, Tuple
from config.constants import JUDGE_PROMPT, DEFAULT_MODEL, DEFAULT_SCHEMA_TEXT
from core.ai_client import GroqAI
from core.utils import normalize_sql # type: ignore


def judge_sql(
    question: str,
    expected_sql: str,
    model_sql: str,
    schema_text: str = DEFAULT_SCHEMA_TEXT,
    model: str = DEFAULT_MODEL,
) -> Tuple[bool, str]:
    """
    Uses an LLM judge to compare expected vs model SQL.
    Returns (is_correct, reason).
    """
    ai = GroqAI()

    prompt = f"""{JUDGE_PROMPT}

Schema:
{schema_text}

Question:
{question}

Expected SQL:
{expected_sql}

Model SQL:
{model_sql}
"""
    messages = [{"role": "user", "content": prompt}]
    text, _, err = ai.chat(messages, model=model,
                           temperature=0.0, max_tokens=200)
    if err:
        return False, f"Judge error: {err}"

    # Try strict JSON parse
    try:
        data = json.loads(text)
        return bool(data.get("is_correct", False)), data.get("reason", "")
    except Exception:
        # Fallback: simple heuristic compare
        return normalize_sql(expected_sql) == normalize_sql(model_sql), "Fallback equality check"


def run_eval_dataset(dataset: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    For each sample {question, expected_sql}, generates model SQL and judges it.
    Returns list of results with correctness and reason.
    """
    from config.constants import DEFAULT_MODEL
    from core.prompts import build_few_shot_prompt
    ai = GroqAI()
    results = []

    for sample in dataset:
        q = sample["question"]
        expected = sample["expected_sql"]

        # Use few-shot for stronger baseline
        messages = build_few_shot_prompt(q, schema_text=DEFAULT_SCHEMA_TEXT)
        model_sql, usage, err = ai.chat(
            messages, model=DEFAULT_MODEL, temperature=0.0, max_tokens=200)
        if err:
            results.append({"question": q, "expected_sql": expected,
                           "model_sql": "", "is_correct": False, "reason": err})
            continue

        model_sql_clean = normalize_sql(model_sql)
        expected_clean = normalize_sql(expected)

        # Quick exact compare first
        if model_sql_clean == expected_clean:
            results.append({"question": q, "expected_sql": expected,
                           "model_sql": model_sql, "is_correct": True, "reason": "Exact match"})
        else:
            ok, reason = judge_sql(q, expected, model_sql)
            results.append({"question": q, "expected_sql": expected,
                           "model_sql": model_sql, "is_correct": ok, "reason": reason})

    return results
