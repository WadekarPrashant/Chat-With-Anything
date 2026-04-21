import json
import types
from typing import Dict, List

import streamlit as st
from pymongo import MongoClient
from langchain_core.messages import SystemMessage


@st.cache_resource
def get_mongo_client(host: str = "localhost", port: int = 27017,
                     username: str = "", password: str = "") -> MongoClient:
    """Create and cache a MongoDB client. Cache key includes connection params."""
    if username and password:
        uri = f"mongodb://{username}:{password}@{host}:{port}/"
    else:
        uri = f"mongodb://{host}:{port}/"
    return MongoClient(uri, serverSelectionTimeoutMS=5000)


def get_collection_schema(collection, sample_size: int = 10) -> Dict[str, List[str]]:
    """Infer schema by sampling documents and collecting field types."""
    pipeline = [{"$sample": {"size": sample_size}}, {"$project": {"_id": 0}}]
    samples = list(collection.aggregate(pipeline))
    fields: set = set()
    for doc in samples:
        fields.update(doc.keys())
    schema: Dict[str, List[str]] = {}
    for field in fields:
        field_types: set = set()
        for doc in samples:
            if field in doc:
                field_types.add(type(doc[field]).__name__)
        schema[field] = list(field_types)
    return schema


def generate_mongo_examples(llm, schema: Dict[str, List[str]], n: int = 3) -> List[Dict]:
    """Ask the LLM to generate example NL questions and their MongoDB query filters."""
    schema_str = "\n".join(f"- {f}: {', '.join(t)}" for f, t in schema.items())
    prompt = (
        "You are an expert in MongoDB queries. "
        "Given a MongoDB collection with the following schema:\n\n"
        f"{schema_str}\n\n"
        f"Generate {n} diverse examples: (1) a natural-language question a user might ask, "
        "and (2) the corresponding MongoDB query filter as a JSON object. "
        "Return ONLY a JSON array — no extra text or markdown:\n\n"
        '[{"question": "…", "query": {…}}, …]\n\n'
        "Include examples of filtering, comparisons, and logical operators where applicable."
    )
    response = llm.invoke([SystemMessage(content=prompt)])
    try:
        content = _strip_fences(response.content)
        examples = json.loads(content)
        for ex in examples:
            if isinstance(ex.get("query"), str):
                try:
                    ex["query"] = json.loads(ex["query"])
                except Exception:
                    pass
        return examples
    except Exception as e:
        st.warning(f"Failed to generate MongoDB examples: {e}")
        return [
            {"question": "Find all documents", "query": {}},
            {"question": "Count total documents", "query": {}},
        ]


def get_mongo_query(llm, schema: Dict[str, List[str]],
                    examples: List[Dict], user_question: str) -> Dict:
    """Translate a natural-language question into a MongoDB query filter dict."""
    schema_str = "\n".join(f"- {f}: {', '.join(t)}" for f, t in schema.items())
    examples_str = "\n\n".join(
        f"Question: {ex['question']}\nQuery: {json.dumps(ex['query'])}"
        for ex in examples
    )
    prompt = (
        "You are a helpful assistant that translates natural language questions into MongoDB query filters.\n\n"
        f"Collection schema:\n{schema_str}\n\n"
        f"Examples:\n{examples_str}\n\n"
        f'Translate this question: "{user_question}"\n\n'
        "Return ONLY a valid JSON object — no markdown, no extra text."
    )
    response = llm.invoke([SystemMessage(content=prompt)])
    content = _strip_fences(response.content)
    return json.loads(content)


def display_mongo_examples(examples: List[Dict]) -> None:
    """Render MongoDB example questions in the Streamlit UI."""
    st.subheader("Example Questions")
    for i, ex in enumerate(examples):
        with st.expander(f"Example {i + 1}: {ex['question']}"):
            query_display = (
                json.dumps(ex["query"], indent=2)
                if isinstance(ex["query"], dict)
                else str(ex["query"])
            )
            st.code(query_display, language="json")


def _strip_fences(text: str) -> str:
    """Remove markdown code fences from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        # parts[1] is the content between the first and second ```
        text = parts[1]
        # Drop a leading language tag (e.g. "json\n")
        if "\n" in text:
            first_line, rest = text.split("\n", 1)
            if first_line.strip().isalpha():
                text = rest
    return text.replace("`", "").strip()
