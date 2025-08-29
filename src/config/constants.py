APP_TITLE = "🧠  AI Chat → SQL Generator"

# Groq model (fast and capable)
DEFAULT_MODEL = "llama-3.1-8b-instant"

# App modes
MODES = ["Zero-Shot", "One-Shot", "Few-Shot"]

# Stop sequences to keep output clean
STOP_SEQUENCES_DEFAULT = ["```", "\n\n--", "\n#"]

# Sample SQLite DB path
DEFAULT_DB_PATH = "data/sample.db"

# Default schema injected into prompts (aligns with sample.db)
DEFAULT_SCHEMA_TEXT = """
Database Schema (SQLite):

Table: customers
- id INTEGER PRIMARY KEY
- name TEXT
- email TEXT
- created_at TEXT

Table: orders
- id INTEGER PRIMARY KEY
- customer_id INTEGER (FK → customers.id)
- amount REAL
- created_at TEXT

Table: products
- id INTEGER PRIMARY KEY
- name TEXT
- price REAL
- stock INTEGER
"""

# Judge prompt used in evaluation (phase for tests/evaluator)
JUDGE_PROMPT = """
You are a careful SQL judge. Given a database schema, a natural language question,
an expected SQL query, and a model-generated SQL query, decide if the model query is
correct and answers the question for the given schema. Consider logical equivalence
(e.g., selected columns vs * where appropriate), conditions, and date intervals.

Respond strictly in JSON with fields:
{"is_correct": true/false, "reason": "<short reason>"}
"""
