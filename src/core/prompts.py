from typing import List, Dict, Optional


SYSTEM_PROMPT = """
You are an intelligent SQL generator.
- Output ONLY SQL (PostgreSQL/SQLite compatible).
- Do NOT include explanations or backticks.
- Use only the columns and tables from the provided schema.
- Prefer ISO date strings for comparisons when needed.
"""


def _system_msg(schema_text: str) -> Dict[str, str]:
    return {"role": "system", "content": SYSTEM_PROMPT.strip() + "\n\n" + schema_text.strip()}


def _user_q_prompt(user_query: str) -> str:
    return f"Question: {user_query}\nSQL:"


def build_zero_shot_prompt(user_query: str, schema_text: str, stop_sequences: Optional[list] = None) -> List[Dict[str, str]]:
    return [
        _system_msg(schema_text),
        {"role": "user", "content": _user_q_prompt(user_query)},
    ]


def build_one_shot_prompt(user_query: str, schema_text: str, stop_sequences: Optional[list] = None) -> List[Dict[str, str]]:
    example = (
        "Question: List all customers created in the last 30 days\n"
        "SQL: SELECT * FROM customers WHERE date(created_at) >= date('now','-30 day');"
    )
    return [
        _system_msg(schema_text),
        {"role": "user", "content": example},
        {"role": "user", "content": _user_q_prompt(user_query)},
    ]


def build_few_shot_prompt(user_query: str, schema_text: str, stop_sequences: Optional[list] = None) -> List[Dict[str, str]]:
    examples = (
        "Question: List orders with amount greater than 200\n"
        "SQL: SELECT * FROM orders WHERE amount > 200;\n\n"
        "Question: Get product names where stock is less than 10\n"
        "SQL: SELECT name FROM products WHERE stock < 10;"
    )
    return [
        _system_msg(schema_text),
        {"role": "user", "content": examples},
        {"role": "user", "content": _user_q_prompt(user_query)},
    ]