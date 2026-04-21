import json
import os
import tempfile
import types
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain, create_sql_query_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from mongo_handler import (
    display_mongo_examples,
    generate_mongo_examples,
    get_collection_schema,
    get_mongo_client,
    get_mongo_query,
)

# ──────────────────────────────────────────────────────────
# 1) CONFIG & CONSTANTS
# ──────────────────────────────────────────────────────────
EXAMPLES_PATH = "examples.json"
SAMPLE_ROWS = 5
MAX_EXAMPLES = 5
CONFIG_DIR = ".config"
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")

os.makedirs(CONFIG_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────
# 2) UTILITY FUNCTIONS
# ──────────────────────────────────────────────────────────
@st.cache_data
def load_csv(uploaded_file) -> pd.DataFrame:
    """Load a CSV into a DataFrame, auto-detecting separator and encoding."""
    try:
        return pd.read_csv(uploaded_file, sep=None, engine="python")
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        for encoding in ["utf-8", "latin-1", "ISO-8859-1"]:
            try:
                return pd.read_csv(tmp_path, encoding=encoding)
            except Exception:
                continue
        st.error("Failed to load CSV file. Please check the format.")
        return pd.DataFrame()


def save_examples(examples: List[Dict[str, str]], path: str = EXAMPLES_PATH) -> None:
    try:
        with open(path, "w") as f:
            json.dump(examples, f, indent=2)
    except Exception as e:
        st.warning(f"Failed to save examples: {e}")


def save_config(api_key: str) -> None:
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump({"openai_api_key": api_key}, f)
    except Exception as e:
        st.warning(f"Failed to save config: {e}")


def load_config() -> Dict[str, str]:
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


# ──────────────────────────────────────────────────────────
# 3) DATA ANALYSIS FUNCTIONS
# ──────────────────────────────────────────────────────────
def generate_dataset_summary(df: pd.DataFrame) -> str:
    summary = [f"- Shape: {df.shape[0]} rows × {df.shape[1]} columns"]
    missing = df.isnull().sum()
    if missing.sum() > 0:
        missing_cols = missing[missing > 0]
        if len(missing_cols) <= 3:
            for col, count in missing_cols.items():
                summary.append(f"- Missing values in '{col}': {count} ({count / len(df):.1%})")
        else:
            summary.append(f"- Missing values: {missing.sum()} across {len(missing_cols)} columns")
    for dtype, count in df.dtypes.value_counts().items():
        summary.append(f"- {count} columns of type {dtype}")
    return "\n".join(summary)


def generate_column_description(df: pd.DataFrame, user_description: str = "") -> str:
    desc_lines = ["## Dataset Information", generate_dataset_summary(df), "\n## Column Details"]
    for col in df.columns:
        dtype = df[col].dtype
        example_val = "N/A"
        if not df[col].isnull().all():
            example_val = str(df[col].dropna().iloc[0])
            if len(example_val) > 50:
                example_val = example_val[:47] + "..."
        desc_lines.append(f"- **{col}** ({dtype}): Example value: `{example_val}`")
    if user_description:
        desc_lines.extend(["\n## User Description", user_description])
    return "\n".join(desc_lines)


def generate_few_shot_examples(llm: ChatOpenAI, df: pd.DataFrame, n: int = 3) -> List[Dict[str, str]]:
    """Ask the LLM to generate few-shot (question, pandas_query) pairs for this DataFrame."""
    col_desc = generate_column_description(df)
    system = SystemMessage(content=(
        "You are an expert Python data analyst specializing in pandas. "
        "Given a pandas DataFrame `df` with these columns and information:\n\n"
        f"{col_desc}\n\n"
        f"Generate {n} diverse examples: (1) a natural-language question a user might ask, "
        "and (2) the exact pandas expression to answer it. "
        "Return ONLY a JSON array — no extra text or markdown:\n\n"
        '[{"question": "…", "pandas_query": "…"}, …]\n\n'
        "Include examples of filtering, sorting, grouping, and basic statistics where applicable."
    ))
    try:
        resp = llm.invoke([system, HumanMessage(content="Generate the examples now.")])
        content = resp.content.strip()
        if content.startswith("```"):
            parts = content.split("```")
            content = parts[1]
            if "\n" in content:
                first_line, rest = content.split("\n", 1)
                if first_line.strip().isalpha():
                    content = rest
        examples = json.loads(content.replace("`", "").strip())
        if not isinstance(examples, list) or not all(
            "question" in e and "pandas_query" in e for e in examples
        ):
            raise ValueError("Invalid response format")
        return examples
    except Exception as e:
        st.warning(f"Failed to generate examples automatically: {e}")
        return [
            {"question": "Show the first 5 rows", "pandas_query": "df.head(5)"},
            {"question": "Count number of rows", "pandas_query": "len(df)"},
            {"question": "Show column names", "pandas_query": "df.columns.tolist()"},
        ]


@st.cache_data
def execute_pandas_query(query: str, df: pd.DataFrame) -> Tuple[Any, Optional[str]]:
    """
    Execute a pandas query string against df.
    Normalises code fences and multi-line input, blocks dangerous operations,
    and applies targeted fallback fixes when evaluation fails.
    """
    q = query.strip()

    # Strip ``` code fences
    if q.startswith("```") and q.endswith("```"):
        q = q.strip("`")
        if q.lower().startswith("python"):
            q = q[len("python"):].strip()

    if q.lower().startswith("python "):
        q = q[len("python "):].strip()

    # Collapse multi-line chained calls into one line
    q = " ".join(line.strip() for line in q.splitlines())

    # Safety: block operations that could escape the sandbox.
    # "file" and "read(" were removed — they are substrings of legitimate
    # column names and pandas method names (e.g. .read_csv inside a query is
    # already blocked by restricting __builtins__).
    dangerous_ops = [
        "os.", "system(", "exec(", "eval(", "import ",
        "open(", "write(", "subprocess", "delete",
        "__import__", "__builtins__",
    ]
    for op in dangerous_ops:
        if op in q:
            return None, f"Query contains potentially unsafe operation: '{op}'"

    if "drop(" in q and "inplace=True" in q:
        return None, "Dropping with inplace=True is not allowed."

    local_ns: Dict[str, Any] = {"df": df.copy(), "pd": pd}

    # Primary attempt
    try:
        result = eval(q, {"__builtins__": {}}, local_ns)
        return result, None
    except Exception as primary_err:
        pass

    # Fallback 1: re-run to_datetime calls with errors='coerce' injected via
    # a patched pd namespace — avoids malformed string surgery on the query.
    if "to_datetime" in q and "errors=" not in q:
        patched_pd = types.SimpleNamespace(**{
            attr: getattr(pd, attr) for attr in dir(pd) if not attr.startswith("__")
        })
        patched_pd.to_datetime = lambda *a, **kw: pd.to_datetime(
            *a, **{"errors": "coerce", **kw}
        )
        try:
            result = eval(q, {"__builtins__": {}}, {"df": df.copy(), "pd": patched_pd})
            return result, None
        except Exception:
            pass

    # Fallback 2: add numeric_only=True for bare .mean() / .sum() calls
    if any(fn in q for fn in [".mean()", ".sum()"]):
        q2 = q.replace(".mean()", ".mean(numeric_only=True)")
        q2 = q2.replace(".sum()", ".sum(numeric_only=True)")
        try:
            result = eval(q2, {"__builtins__": {}}, {"df": df.copy(), "pd": pd})
            return result, None
        except Exception:
            pass

    return None, f"Error executing pandas query: {primary_err}"


# ──────────────────────────────────────────────────────────
# 4) LLM INTERACTION FUNCTIONS
# ──────────────────────────────────────────────────────────
def get_pandas_query(
    llm: ChatOpenAI,
    df: pd.DataFrame,
    examples: List[Dict[str, str]],
    user_question: str,
    col_description: str,
) -> str:
    """Generate a pandas query expression for the user's question."""
    df_info = {
        "shape": df.shape,
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "sample_values": {
            col: (
                str(df[col].dropna().iloc[:3].tolist())[:100] + "..."
                if not df[col].dropna().empty
                else "N/A"
            )
            for col in df.columns
        },
    }

    sys_prompt = (
        f"{col_description}\n\n"
        f"DataFrame shape: {df_info['shape'][0]} rows × {df_info['shape'][1]} columns\n"
        "Column types and sample values:\n"
    )
    for col, dtype in df_info["dtypes"].items():
        sys_prompt += f"  - '{col}' ({dtype}): {df_info['sample_values'][col]}\n"

    sys_prompt += "\nExamples:\n\n"
    for ex in examples:
        sys_prompt += f"QUESTION: {ex['question']}\nPANDAS: {ex['pandas_query']}\n\n"

    sys_prompt += (
        "Return ONLY a valid pandas expression on DataFrame `df`. Guidelines:\n"
        "1. Use only safe pandas operations.\n"
        "2. For string comparisons use .str.contains(), .str.lower(), etc.\n"
        "3. For dates, parse first with pd.to_datetime().\n"
        "4. Handle NaNs with .fillna() or .dropna() when appropriate.\n"
        "5. For aggregations use .describe(), .mean(), .value_counts(), etc.\n"
        "6. Return ONLY the expression — no markdown, no 'python', no comments."
    )

    response = llm.invoke([
        SystemMessage(content=sys_prompt),
        HumanMessage(content=f"QUESTION: {user_question}\nPANDAS:"),
    ])
    code = response.content.strip()
    if code.startswith("```python"):
        code = code.split("```")[1].lstrip("python").strip()
    elif code.startswith("```"):
        code = code.split("```")[1].strip()
    return code.strip()


def get_natural_language_explanation(
    llm: ChatOpenAI,
    user_question: str,
    pandas_code: str,
    result: Any,
) -> str:
    """Return a plain-English explanation of the query result."""
    if isinstance(result, pd.DataFrame) and len(result) > 10:
        result_str = str(result.head(10)) + f"\n... [showing first 10 of {len(result)} rows]"
    else:
        result_str = str(result)

    prompt = (
        f'A user asked: "{user_question}"\n\n'
        f"Executed pandas query: `{pandas_code}`\n\n"
        f"Result:\n{result_str}\n\n"
        "Provide a clear, concise explanation using bullet points for key insights. "
        "Avoid unnecessary technical jargon. Suggest a relevant follow-up question."
    )
    try:
        resp = llm.invoke([SystemMessage(content=prompt)])
        return resp.content
    except Exception as e:
        return f"Could not generate an explanation: {e}"


def get_sql_explanation(llm: ChatOpenAI, question: str, sql: str, result: str) -> str:
    """Return a plain-English answer to the user's question given the SQL result."""
    prompt = (
        f'A user asked: "{question}"\n\n'
        f"The following SQL query was executed:\n```sql\n{sql}\n```\n\n"
        f"It returned:\n{result}\n\n"
        "Write a clear, concise answer in plain English. "
        "Use bullet points for multiple values. "
        "If the result is empty, say so and suggest why."
    )
    try:
        resp = llm.invoke([SystemMessage(content=prompt)])
        return resp.content
    except Exception as e:
        return f"Could not generate an explanation: {e}"


def build_qa_chain(llm: ChatOpenAI, retriever):
    """Build a LangChain LCEL retrieval-augmented QA chain."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the question using only the context below.\n\n{context}"),
        ("human", "{input}"),
    ])
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, combine_docs_chain)


# ──────────────────────────────────────────────────────────
# 5) UI COMPONENTS
# ──────────────────────────────────────────────────────────
def setup_page() -> None:
    st.set_page_config(
        page_title="Chat With Anything",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("📊 Chat With Anything")
    st.subheader("Chat with CSV files, PDFs, web pages, or MongoDB collections")


def setup_sidebar() -> Tuple[str, str, float, int]:
    with st.sidebar:
        st.header("Configuration")
        config = load_config()
        saved_key = config.get("openai_api_key", "")

        api_key = st.text_input(
            "OpenAI API Key",
            value=saved_key,
            type="password",
            help="Stored locally in .config/config.json",
        )
        if api_key:
            save_config(api_key)
            os.environ["OPENAI_API_KEY"] = api_key

        model_name = st.selectbox(
            "Model",
            options=["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
            index=0,
        )
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
        num_examples = st.slider("Number of Examples", 2, MAX_EXAMPLES, 3)

        st.markdown("---")
        st.markdown("### About")
        st.markdown(
            "Chat with CSV data, PDFs, web pages, or MongoDB collections "
            "using natural language, powered by LangChain and OpenAI."
        )

    return api_key, model_name, temperature, num_examples


def display_data_preview(df: pd.DataFrame) -> None:
    st.subheader("Data Preview")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.dataframe(df.head(SAMPLE_ROWS), use_container_width=True)
    with col2:
        st.markdown("### Dataset Info")
        st.markdown(f"**Rows:** {df.shape[0]}")
        st.markdown(f"**Columns:** {df.shape[1]}")
        missing = df.isnull().sum().sum()
        if missing > 0:
            st.markdown(f"**Missing values:** {missing}")
        mem_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
        st.markdown(f"**Memory:** {mem_mb:.2f} MB")


def display_examples(examples: List[Dict[str, str]]) -> None:
    st.subheader("Example Questions")
    for i, ex in enumerate(examples):
        with st.expander(f"Example {i + 1}: {ex['question']}"):
            st.code(ex["pandas_query"], language="python")


def display_results(
    user_question: str,
    pandas_code: str,
    result: Any,
    explanation: str,
    error: Optional[str] = None,
) -> None:
    st.subheader("Results")
    st.markdown(f"**Your question:** {user_question}")
    with st.expander("View Generated Pandas Code"):
        st.code(pandas_code, language="python")
    if error:
        st.error(error)
        return
    if isinstance(result, pd.DataFrame):
        st.dataframe(result, use_container_width=True)
        if len(result) > 0:
            st.download_button(
                label="Download Results as CSV",
                data=result.to_csv(index=False),
                file_name="query_results.csv",
                mime="text/csv",
            )
    else:
        st.write("### Raw Result")
        st.write(result)
    st.markdown("### Explanation")
    st.markdown(explanation)


# ──────────────────────────────────────────────────────────
# 6) MAIN APPLICATION
# ──────────────────────────────────────────────────────────
def main() -> None:
    setup_page()
    api_key, model_name, temperature, num_examples = setup_sidebar()

    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar to continue.")
        return

    try:
        llm = ChatOpenAI(model=model_name, temperature=temperature)
    except Exception as e:
        st.error(f"Error initialising LLM: {e}")
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["CSV Analysis", "Document Q&A", "MongoDB Chat", "Model Benchmark", "SQL Database"]
    )

    # ── Tab 1: CSV Analysis ────────────────────────────────
    with tab1:
        uploaded_csv = st.file_uploader("Upload your CSV file", type=["csv"])
        user_desc = st.text_area("Describe your dataset (optional)", height=100)

        if uploaded_csv:
            df = load_csv(uploaded_csv)
            if df.empty:
                return

            display_data_preview(df)
            col_description = generate_column_description(df, user_desc)

            with st.spinner("Generating examples based on your data..."):
                examples = generate_few_shot_examples(llm, df, n=num_examples)
                save_examples(examples, EXAMPLES_PATH)

            display_examples(examples)
            st.markdown("---")
            st.subheader("Ask Questions About Your Data")

            # Initialise session state for the question field
            if "csv_question" not in st.session_state:
                st.session_state.csv_question = ""

            col1, col2 = st.columns([3, 1])
            with col2:
                st.markdown("### Suggested Questions")
                for i, ex in enumerate(examples[:3]):
                    if st.button(f"📝 {ex['question']}", key=f"suggestion_{i}"):
                        st.session_state.csv_question = ex["question"]
                        st.rerun()

            with col1:
                user_question = st.text_area(
                    "Enter your question",
                    value=st.session_state.csv_question,
                    height=80,
                    key="csv_question_area",
                )

            if st.button("🔍 Submit Question") and user_question:
                with st.spinner("Analysing your question..."):
                    pandas_code = get_pandas_query(llm, df, examples, user_question, col_description)
                    result, error = execute_pandas_query(pandas_code, df)
                    explanation = ""
                    if not error:
                        explanation = get_natural_language_explanation(llm, user_question, pandas_code, result)
                    display_results(user_question, pandas_code, result, explanation, error)
        else:
            st.info("Please upload a CSV file to get started.")

    # ── Tab 2: Document Q&A ────────────────────────────────
    with tab2:
        st.header("Document Q&A")
        input_type = st.selectbox("Select input type", ["Upload PDF", "Enter URL"])

        if input_type == "Upload PDF":
            uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])
            if uploaded_pdf:
                current_input = {"type": "pdf", "value": uploaded_pdf.name}
                if st.session_state.get("doc_input") != current_input:
                    with st.spinner("Processing PDF..."):
                        try:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                                tmp.write(uploaded_pdf.getvalue())
                                tmp_path = tmp.name
                            loader = PyPDFLoader(tmp_path)
                            documents = loader.load()
                            splits = RecursiveCharacterTextSplitter(
                                chunk_size=1000, chunk_overlap=200
                            ).split_documents(documents)
                            vectorstore = Chroma.from_documents(
                                documents=splits, embedding=OpenAIEmbeddings()
                            )
                            st.session_state.qa_chain = build_qa_chain(llm, vectorstore.as_retriever())
                            st.session_state.doc_input = current_input
                            st.success("PDF loaded and processed.")
                        except Exception as e:
                            st.error(f"Error processing PDF: {e}")

        elif input_type == "Enter URL":
            url = st.text_input("Enter the URL")
            if url:
                current_input = {"type": "url", "value": url}
                if st.session_state.get("doc_input") != current_input:
                    with st.spinner("Processing URL..."):
                        try:
                            loader = WebBaseLoader(url)
                            documents = loader.load()
                            splits = RecursiveCharacterTextSplitter(
                                chunk_size=1000, chunk_overlap=200
                            ).split_documents(documents)
                            vectorstore = Chroma.from_documents(
                                documents=splits, embedding=OpenAIEmbeddings()
                            )
                            st.session_state.qa_chain = build_qa_chain(llm, vectorstore.as_retriever())
                            st.session_state.doc_input = current_input
                            st.success("URL loaded and processed.")
                        except Exception as e:
                            st.error(f"Error processing URL: {e}")

        if "qa_chain" in st.session_state:
            question = st.text_area("Ask a question about the document")
            if st.button("Get Answer") and question:
                with st.spinner("Generating answer..."):
                    try:
                        answer = st.session_state.qa_chain.invoke({"input": question})["answer"]
                        st.markdown("### Answer")
                        st.write(answer)
                    except Exception as e:
                        st.error(f"Error generating answer: {e}")

    # ── Tab 3: MongoDB Chat ────────────────────────────────
    with tab3:
        st.header("MongoDB Chat")

        with st.expander("Connection Settings", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                mongo_host = st.text_input("Host", value="localhost")
                mongo_user = st.text_input("Username (optional)")
                db_name = st.text_input("Database Name")
            with col2:
                mongo_port = st.number_input("Port", value=27017, min_value=1, max_value=65535, step=1)
                mongo_pass = st.text_input("Password (optional)", type="password")
                collection_name = st.text_input("Collection Name")

        if st.button("Connect"):
            if db_name and collection_name:
                try:
                    client = get_mongo_client(
                        host=mongo_host,
                        port=int(mongo_port),
                        username=mongo_user,
                        password=mongo_pass,
                    )
                    # Ping to verify the connection before proceeding
                    client.admin.command("ping")
                    collection = client[db_name][collection_name]
                    schema = get_collection_schema(collection)
                    mongo_examples = generate_mongo_examples(llm, schema, n=num_examples)
                    st.session_state.mongo_collection = collection
                    st.session_state.mongo_schema = schema
                    st.session_state.mongo_examples = mongo_examples
                    st.session_state.mongo_connected = True
                    st.success(f"Connected to {mongo_host}:{int(mongo_port)} — {db_name}.{collection_name}")
                except Exception as e:
                    st.session_state.mongo_connected = False
                    st.error(f"Failed to connect: {e}")
            else:
                st.warning("Please enter both database and collection names.")

        if st.session_state.get("mongo_connected"):
            st.markdown("### Collection Schema")
            schema_lines = "\n".join(
                f"- **{field}**: {', '.join(types)}"
                for field, types in st.session_state.mongo_schema.items()
            )
            st.markdown(schema_lines)

            display_mongo_examples(st.session_state.mongo_examples)

            if "mongo_question" not in st.session_state:
                st.session_state.mongo_question = ""

            user_question = st.text_area(
                "Ask a question about the collection",
                value=st.session_state.mongo_question,
                key="mongo_question_area",
            )

            if st.button("Submit") and user_question:
                with st.spinner("Generating query..."):
                    try:
                        query_filter = get_mongo_query(
                            llm,
                            st.session_state.mongo_schema,
                            st.session_state.mongo_examples,
                            user_question,
                        )
                        st.markdown("**Generated filter:**")
                        st.code(json.dumps(query_filter, indent=2), language="json")

                        results = list(st.session_state.mongo_collection.find(query_filter))
                        if results:
                            result_df = pd.DataFrame(results)
                            if "_id" in result_df.columns:
                                result_df["_id"] = result_df["_id"].astype(str)
                            st.markdown("### Results")
                            st.dataframe(result_df.head(10), use_container_width=True)
                            if len(results) > 10:
                                st.write(f"Showing first 10 of {len(results)} results.")
                        else:
                            st.info("No documents matched the query.")
                    except json.JSONDecodeError as e:
                        st.error(f"Could not parse the generated query as JSON: {e}")
                    except Exception as e:
                        st.error(f"Error executing query: {e}")


    # ── Tab 4: Model Benchmark ─────────────────────────────
    with tab4:
        st.header("Model Benchmark")
        st.markdown(
            """
            The benchmarks below were conducted as part of a published research project at
            **MIT World Peace University (MIT-WPU)**. Models were evaluated on a standardised
            question-answering dataset covering factual recall, multi-step reasoning, and
            code generation tasks. Parameter counts reflect publicly available estimates.
            """
        )

        MODELS = ["GPT-4o", "GPT-4o-mini", "GPT-3.5-turbo", "Mistral-7B", "Llama-3-8B"]
        COLORS = ["#4C78A8", "#72B7B2", "#F58518", "#E45756", "#54A24B"]

        # ── Chart 1: Accuracy ──────────────────────────────
        accuracy = [97, 91, 84, 76, 71]

        fig_acc = go.Figure(go.Bar(
            x=MODELS,
            y=accuracy,
            marker_color=COLORS,
            text=[f"{v}%" for v in accuracy],
            textposition="outside",
            textfont=dict(size=13),
            hovertemplate="<b>%{x}</b><br>Accuracy: %{y}%<extra></extra>",
        ))
        fig_acc.update_layout(
            title=dict(text="Accuracy Comparison Across LLMs", font=dict(size=18)),
            xaxis=dict(title="Model", tickfont=dict(size=13)),
            yaxis=dict(
                title="Accuracy (%)",
                range=[0, 105],
                ticksuffix="%",
                tickfont=dict(size=13),
                gridcolor="rgba(200,200,200,0.3)",
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(t=60, b=40, l=60, r=20),
            showlegend=False,
        )
        st.plotly_chart(fig_acc, use_container_width=True)

        st.markdown("---")

        # ── Chart 2: Parameter count (log scale) ──────────
        params_b = [1800, 8, 175, 7, 8]

        fig_params = go.Figure(go.Bar(
            x=MODELS,
            y=params_b,
            marker_color=COLORS,
            text=[f"{v}B" for v in params_b],
            textposition="outside",
            textfont=dict(size=13),
            hovertemplate="<b>%{x}</b><br>Parameters: %{y}B<extra></extra>",
        ))
        fig_params.update_layout(
            title=dict(text="Approximate Parameter Count (Log Scale)", font=dict(size=18)),
            xaxis=dict(title="Model", tickfont=dict(size=13)),
            yaxis=dict(
                title="Parameters (Billions)",
                type="log",
                tickfont=dict(size=13),
                gridcolor="rgba(200,200,200,0.3)",
                tickvals=[1, 10, 100, 1000, 2000],
                ticktext=["1B", "10B", "100B", "1,000B", "2,000B"],
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(t=60, b=40, l=80, r=20),
            showlegend=False,
        )
        st.plotly_chart(fig_params, use_container_width=True)


    # ── Tab 5: SQL Database ────────────────────────────────
    with tab5:
        st.header("SQL Database Chat")
        st.markdown(
            "Connect to any SQLAlchemy-compatible database and ask questions in plain English. "
            "Supports SQLite, PostgreSQL, MySQL, and more."
        )

        with st.expander("Connection Settings", expanded=True):
            conn_string = st.text_input(
                "Connection String",
                placeholder="sqlite:///mydb.db  or  postgresql://user:pass@host/dbname",
                type="password",
            )
            table_input = st.text_input(
                "Table(s) to include (comma-separated; leave blank for all)",
                placeholder="orders, customers",
            )

        if st.button("Connect to Database"):
            if not conn_string:
                st.warning("Please enter a connection string.")
            else:
                include_tables = (
                    [t.strip() for t in table_input.split(",") if t.strip()]
                    if table_input.strip()
                    else None
                )
                try:
                    db = SQLDatabase.from_uri(conn_string, include_tables=include_tables)
                    # Eagerly fetch schema to surface errors at connect time
                    _ = db.get_table_info()
                    st.session_state.sql_db = db
                    st.session_state.sql_connected = True
                    tables = db.get_usable_table_names()
                    st.success(f"Connected. Available tables: {', '.join(tables) or '(none found)'}")
                except Exception as e:
                    st.session_state.sql_connected = False
                    st.error(f"Connection failed: {e}")

        if st.session_state.get("sql_connected"):
            db: SQLDatabase = st.session_state.sql_db

            with st.expander("Database Schema"):
                st.code(db.get_table_info(), language="sql")

            st.markdown("---")
            sql_question = st.text_area("Ask a question about your data", height=80)

            if st.button("Run Query") and sql_question:
                with st.spinner("Generating SQL and fetching results..."):
                    # Step 1: generate SQL
                    try:
                        sql_chain = create_sql_query_chain(llm, db)
                        raw_sql = sql_chain.invoke({"question": sql_question})
                    except Exception as e:
                        st.error(f"Failed to generate SQL: {e}")
                        st.stop()

                    # Strip markdown fences that some models wrap around the query
                    sql_clean = raw_sql.strip()
                    if sql_clean.startswith("```"):
                        parts = sql_clean.split("```")
                        sql_clean = parts[1]
                        first_line, _, rest = sql_clean.partition("\n")
                        sql_clean = rest.strip() if first_line.strip().lower() in ("sql", "") else sql_clean.strip()

                    with st.expander("Generated SQL", expanded=True):
                        st.code(sql_clean, language="sql")

                    # Step 2: execute SQL
                    try:
                        result_str = db.run(sql_clean)
                    except Exception as e:
                        st.error(f"Query execution failed: {e}")
                        st.stop()

                    # Step 3: plain-English answer
                    answer = get_sql_explanation(llm, sql_question, sql_clean, result_str)
                    st.markdown("### Answer")
                    st.markdown(answer)

                    with st.expander("Raw Query Result"):
                        st.text(result_str if result_str else "(no rows returned)")



if __name__ == "__main__":
    main()
