import os
import json
import tempfile
from typing import List, Dict, Any, Tuple, Optional

import streamlit as st
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# —––––––––––––––––––––––––––––––––––––––––––––––––––––––—
# 1) CONFIG & CONSTANTS
# —––––––––––––––––––––––––––––––––––––––––––––––––––––––—
EXAMPLES_PATH = "examples.json"
SAMPLE_ROWS = 5  # Number of rows to display in preview
MAX_EXAMPLES = 5  # Maximum number of examples to generate
CONFIG_DIR = ".config"
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")

# Create config directory if it doesn't exist
os.makedirs(CONFIG_DIR, exist_ok=True)

# —––––––––––––––––––––––––––––––––––––––––––––––––––––––—
# 2) UTILITY FUNCTIONS
# —––––––––––––––––––––––––––––––––––––––––––––––––––––––—
@st.cache_data
def load_csv(uploaded_file) -> pd.DataFrame:
    """Load CSV into a pandas DataFrame with error handling."""
    try:
        # First try to infer the separator
        return pd.read_csv(uploaded_file, sep=None, engine='python')
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        # Save the uploaded file to a temporary location so we can try different approaches
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        # Try with different encodings
        for encoding in ['utf-8', 'latin-1', 'ISO-8859-1']:
            try:
                return pd.read_csv(tmp_path, encoding=encoding)
            except:
                continue
                
        # If all else fails, return an empty dataframe
        st.error("Failed to load CSV file. Please check the format.")
        return pd.DataFrame()

def load_examples(path: str = EXAMPLES_PATH) -> List[Dict[str, str]]:
    """Load few-shot examples from disk (JSON)."""
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                examples = json.load(f)
            if isinstance(examples, list) and all("question" in e and "pandas_query" in e for e in examples):
                return examples
        except Exception:
            pass
    return [
        {"question": "Show the first 5 rows", "pandas_query": "df.head(5)"},
        {"question": "Count number of rows", "pandas_query": "len(df)"}
    ]

def save_examples(examples: List[Dict[str, str]], path: str = EXAMPLES_PATH) -> None:
    """Persist few-shot examples to disk."""
    try:
        with open(path, "w") as f:
            json.dump(examples, f, indent=2)
    except Exception as e:
        st.warning(f"Failed to save examples: {e}")

def save_config(api_key: str) -> None:
    """Save the OpenAI API key to config file."""
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump({"openai_api_key": api_key}, f)
    except Exception as e:
        st.warning(f"Failed to save config: {e}")

def load_config() -> Dict[str, str]:
    """Load configuration from disk."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

# —––––––––––––––––––––––––––––––––––––––––––––––––––––––—
# 3) DATA ANALYSIS FUNCTIONS
# —––––––––––––––––––––––––––––––––––––––––––––––––––––––—
def generate_dataset_summary(df: pd.DataFrame) -> str:
    """Generate a concise summary of the dataset."""
    summary = []
    summary.append(f"- Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Add info about missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        missing_cols = missing[missing > 0]
        if len(missing_cols) <= 3:
            for col, count in missing_cols.items():
                summary.append(f"- Missing values in '{col}': {count} ({count/len(df):.1%})")
        else:
            summary.append(f"- Missing values: {missing.sum()} across {len(missing_cols)} columns")
    
    # Add info about data types
    dtypes = df.dtypes.value_counts()
    for dtype, count in dtypes.items():
        summary.append(f"- {count} columns of type {dtype}")
    
    return "\n".join(summary)

def generate_column_description(df: pd.DataFrame, user_description: str = "") -> str:
    """
    Build a rich description of the DataFrame's columns, including data types,
    example values, and user description.
    """
    desc_lines = ["## Dataset Information"]
    
    # Add summary statistics
    desc_lines.append(generate_dataset_summary(df))
    desc_lines.append("\n## Column Details")
    
    # Add column descriptions with example values
    for col in df.columns:
        dtype = df[col].dtype
        example_val = "N/A"
        if not df[col].isnull().all():
            example_val = str(df[col].dropna().iloc[0])
            if len(example_val) > 50:
                example_val = example_val[:47] + "..."
        
        desc_lines.append(f"- **{col}** ({dtype}): Example value: `{example_val}`")
    
    # Add user description if provided
    if user_description:
        desc_lines.append("\n## User Description")
        desc_lines.append(user_description)
    
    return "\n".join(desc_lines)

def generate_few_shot_examples(
    llm: ChatOpenAI,
    df: pd.DataFrame,
    n: int = 3
) -> List[Dict[str, str]]:
    """
    Ask the LLM to propose few-shot examples for this DataFrame.
    Returns a list of dicts: [{"question": "...", "pandas_query": "..."}, …].
    """
    col_desc = generate_column_description(df)
    system = SystemMessage(content=(
        "You are an expert Python data analyst specializing in pandas. "
        "Given a pandas DataFrame `df` with these columns and information:\n\n"
        f"{col_desc}\n\n"
        f"Please generate {n} diverse examples of (1) a natural-language question "
        "a user might ask about this data, and (2) the exact pandas expression to answer it. "
        "Focus on questions that demonstrate different pandas capabilities and column relationships. "
        "Return only a JSON array like:\n\n"
        "[{\"question\": \"…\", \"pandas_query\": \"…\"}, …]\n\n"
        "The pandas code should be correct, concise and useful. "
        "Include examples of filtering, sorting, grouping, and basic statistics where applicable. "
        "Do NOT include any extra text or commentary."
    ))
    human = HumanMessage(content="Generate the examples now.")
    
    try:
        resp = llm([system, human])
        examples = json.loads(resp.content)
        if not isinstance(examples, list) or not all("question" in e and "pandas_query" in e for e in examples):
            raise ValueError("Invalid response format")
        return examples
    except Exception as e:
        st.warning(f"Failed to generate examples automatically: {e}")
        return [
            {"question": "Show the first 5 rows", "pandas_query": "df.head(5)"},
            {"question": "Count number of rows", "pandas_query": "len(df)"},
            {"question": "Show column names", "pandas_query": "df.columns.tolist()"}
        ]

@st.cache_data
def execute_pandas_query(query: str, df: pd.DataFrame) -> Tuple[Any, Optional[str]]:
    """
    Execute a pandas query with improved success rate and safety checks.
    Normalizes common user-input patterns (e.g., code fences, 'python' prefixes, multiline dots) 
    and returns the result or an error message.
    """
    # 1. Normalize the query string
    q = query.strip()

    # Remove code fences (``` or ```python ... ```)
    if q.startswith("```") and q.endswith("```"):
        # Strip triple backticks and any leading language identifier
        q = q.strip("`").replace("python", "").strip()

    # Remove leading 'python' keyword if present (case-insensitive)
    if q.lower().startswith("python "):
        q = q[len("python "):].strip()

    # Collapse multi-line chained methods into a single line
    q = " ".join(line.strip() for line in q.splitlines())

    # 2. Safety checks for dangerous operations
    dangerous_ops = [
        "os.", "system(", "exec(", "eval(", "import ",
        "open(", "write(", "read(", "file", "subprocess", 
        "request", "delete"
    ]
    for op in dangerous_ops:
        if op in q:
            return None, f"Query contains potentially unsafe operation: {op}"

    # Prevent in-place drops
    if "drop(" in q and "inplace=True" in q:
        return None, "Dropping with inplace=True is not allowed as it modifies the original dataframe"

    # 3. Attempt execution
    try:
        local_ns = {"df": df.copy(), "pd": pd}
        result = eval(q, {"__builtins__": {}}, local_ns)
        return result, None

    except Exception as e:
        # Fallback fixes
        try:
            # Handle to_datetime without errors parameter
            if "to_datetime" in q and "errors=" not in q:
                q2 = q.replace("to_datetime", "to_datetime(errors='coerce', ", 1)
                return execute_pandas_query(q2, df)

            # Add numeric_only=True for aggregation
            if any(fn in q for fn in [".mean()", ".sum()"]):
                q2 = q.replace(".mean()", ".mean(numeric_only=True)")
                q2 = q2.replace(".sum()", ".sum(numeric_only=True)")
                return examine_pandas_query(q2, df)

            return None, f"Error executing pandas query: {e}"
        except Exception as e2:
            return None, f"Error executing pandas query: {e2}"

# —––––––––––––––––––––––––––––––––––––––––––––––––––––––—
# 4) LLM INTERACTION FUNCTIONS
# —––––––––––––––––––––––––––––––––––––––––––––––––––––––—
def get_pandas_query(
    llm: ChatOpenAI,
    df: pd.DataFrame,
    examples: List[Dict[str, str]],
    user_question: str,
    col_description: str,
) -> str:
    """Generate a pandas query for the user's question with improved accuracy."""
    # Get dataframe info for more context
    df_info = {
        "shape": df.shape,
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "sample_values": {col: str(df[col].dropna().iloc[:3].tolist())[:100] + "..." 
                         if not df[col].dropna().empty else "N/A" 
                         for col in df.columns}
    }
    
    # Construct the system prompt with examples and detailed info
    sys_prompt = (
        f"{col_description}\n\n"
        "Detailed DataFrame Information:\n"
        f"- Shape: {df_info['shape'][0]} rows × {df_info['shape'][1]} columns\n"
        "- Column Types and Sample Values:\n"
    )
    
    # Add column details
    for col, dtype in df_info['dtypes'].items():
        sys_prompt += f"  - '{col}' ({dtype}): {df_info['sample_values'][col]}\n"
    
    sys_prompt += "\nExamples of questions and their pandas solutions:\n\n"
    
    for ex in examples:
        sys_prompt += f"QUESTION: {ex['question']}\nPANDAS: {ex['pandas_query']}\n\n"
    
    sys_prompt += (
        "Now, answer the user's question by returning ONLY a valid pandas "
        "expression on DataFrame `df`. Follow these guidelines:\n\n"
        "1. Use only pandas operations that are safe and effective\n"
        "2. For string comparisons with text columns, use .str methods like .str.contains(), .str.lower()\n"
        "3. For date operations, ensure proper parsing with pd.to_datetime() first\n"
        "4. Handle potential NaN values with .fillna() or .dropna() when appropriate\n"
        "5. Keep the query concise but complete\n"
        "6. For statistical questions, prefer descriptive methods like .describe(), .mean(), etc.\n"
        "7. For complex calculations, break them into clear steps using variables\n"
        "8. Return ONLY the pandas code - no explanations or comments\n\n"
        "9. Do not return anything like pandas, I will execute whatever you will return to me"
        "e.g. When asked how many students passed this exam, only type 'f['Pass_Fail'].str.lower().value_counts().get('pass', 0)', do not type python on top of it"
    )
    
    # Get the response
    response = llm([
        SystemMessage(content=sys_prompt),
        HumanMessage(content=f"QUESTION: {user_question}\nPANDAS:")
    ])
    
    # Clean up the response to ensure it's valid pandas code
    pandas_code = response.content.strip()
    
    # Remove markdown code blocks if present
    if pandas_code.startswith("```python"):
        pandas_code = pandas_code.split("```")[1]
    elif pandas_code.startswith("```"):
        pandas_code = pandas_code.split("```")[1]
    
    return pandas_code.strip()

def get_natural_language_explanation(
    llm: ChatOpenAI,
    user_question: str,
    pandas_code: str,
    result: Any,
) -> str:
    """Generate a natural language explanation of the results."""
    # Format the result for the prompt
    if isinstance(result, pd.DataFrame) and len(result) > 10:
        result_str = str(result.head(10)) + "\n... [showing first 10 rows of " + str(len(result)) + "]"
    else:
        result_str = str(result)
    
    rag_prompt = (
        f"You are a helpful data analyst assistant. A user asked: \"{user_question}\"\n\n"
        f"I executed this pandas query: `{pandas_code}`\n\n"
        f"It returned this result:\n{result_str}\n\n"
        "Please provide a clear, concise explanation that answers the user's question based on "
        "this result. Use bullet points for key insights and avoid unnecessary technical jargon. "
        "If appropriate, suggest follow-up questions the user might want to ask."
    )
    
    try:
        rag_resp = llm([SystemMessage(content=rag_prompt)])
        return rag_resp.content
    except Exception as e:
        return f"I couldn't generate an explanation due to an error: {e}"

# —––––––––––––––––––––––––––––––––––––––––––––––––––––––—
# 5) UI COMPONENTS
# —––––––––––––––––––––––––––––––––––––––––––––––––––––––—
def setup_page():
    """Configure the Streamlit page."""
    st.set_page_config(
        page_title="RAG-powered CSV Chat",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📊 RAG-powered CSV Chat")
    st.subheader("Upload a CSV file or interact with PDFs and web pages")

def setup_sidebar():
    """Set up the sidebar with configuration options."""
    with st.sidebar:
        st.header("Configuration")
        
        # Load saved API key
        config = load_config()
        saved_key = config.get("openai_api_key", "")
        
        # API key input with masking
        api_key = st.text_input(
            "OpenAI API Key", 
            value=saved_key,
            type="password",
            help="Enter your OpenAI API key. It will be stored locally."
        )
        
        if api_key:
            save_config(api_key)
            os.environ["OPENAI_API_KEY"] = api_key
        
        # Model selection
        model_name = st.selectbox(
            "Model",
            options=["gpt-4o", "gpt-3.5-turbo", "gpt-4o-mini"],
            index=0,
            help="Select the OpenAI model to use"
        )
        
        # Temperature setting
        temperature = st.slider(
            "Temperature", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.0, 
            step=0.1,
            help="Higher values make output more random, lower values more deterministic"
        )
        
        # Number of examples
        num_examples = st.slider(
            "Number of Examples", 
            min_value=2, 
            max_value=MAX_EXAMPLES, 
            value=3,
            help="Number of few-shot examples to generate (for CSV analysis)"
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown(
            "This app allows you to chat with your CSV data, PDFs, or web pages using natural language. "
            "It uses LangChain and OpenAI for processing."
        )
    
    return api_key, model_name, temperature, num_examples

def display_data_preview(df: pd.DataFrame):
    """Display a preview of the loaded data."""
    st.subheader("Data Preview")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.dataframe(df.head(SAMPLE_ROWS), use_container_width=True)
    
    with col2:
        st.markdown("### Dataset Info")
        st.markdown(f"**Rows:** {df.shape[0]}")
        st.markdown(f"**Columns:** {df.shape[1]}")
        
        # Show missing values
        missing = df.isnull().sum().sum()
        if missing > 0:
            st.markdown(f"**Missing values:** {missing}")
        
        # Show memory usage
        memory_usage = df.memory_usage(deep=True).sum()
        st.markdown(f"**Memory usage:** {memory_usage / 1024**2:.2f} MB")

def display_examples(examples: List[Dict[str, str]]):
    """Display the few-shot examples in a structured way."""
    st.subheader("Example Questions")
    
    for i, ex in enumerate(examples):
        with st.expander(f"Example {i+1}: {ex['question']}"):
            st.code(ex['pandas_query'], language="python")

def display_results(
    user_question: str,
    pandas_code: str,
    result: Any,
    explanation: str,
    error: Optional[str] = None
):
    """Display the results of the query in a structured way."""
    st.subheader("Results")
    
    # Display the question
    st.markdown(f"**Your question:** {user_question}")
    
    # Display the pandas code
    with st.expander("View Generated Pandas Code"):
        st.code(pandas_code, language="python")
    
    # Display error if any
    if error:
        st.error(error)
        return
    
    # Display the result
    if isinstance(result, pd.DataFrame):
        st.dataframe(result, use_container_width=True)
        
        # Add download button for DataFrame results
        if len(result) > 0:
            csv = result.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="query_results.csv",
                mime="text/csv"
            )
    else:
        st.write("### Raw Result")
        st.write(result)
    
    # Display the explanation
    st.markdown("### Explanation")
    st.markdown(explanation)

# —––––––––––––––––––––––––––––––––––––––––––––––––––––––—
# 6) MAIN APPLICATION
# —––––––––––––––––––––––––––––––––––––––––––––––––––––––—
def main():
    """Main application function."""
    setup_page()
    api_key, model_name, temperature, num_examples = setup_sidebar()
    
    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar to continue.")
        return
    
    # Initialize LLM
    try:
        llm = ChatOpenAI(model_name=model_name, temperature=temperature)
    except Exception as e:
        st.error(f"Error initializing LLM: {e}")
        return
    
    # Create tabs for CSV and Document Q&A
    tab1, tab2 = st.tabs(["CSV Analysis", "Document Q&A"])
    
    # CSV Analysis Tab
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
            
            col1, col2 = st.columns([3, 1])
            with col1:
                user_question = st.text_area("Enter your question", height=80)
            with col2:
                st.markdown("### Suggested Questions")
                for i, ex in enumerate(examples[:3]):
                    if st.button(f"📝 {ex['question']}", key=f"suggestion_{i}"):
                        user_question = ex['question']
                        st.session_state.user_question = user_question
            
            submit_button = st.button("🔍 Submit Question")
            
            if submit_button and user_question:
                with st.spinner("Analyzing your question..."):
                    pandas_code = get_pandas_query(llm, df, examples, user_question, col_description)
                    result, error = execute_pandas_query(pandas_code, df)
                    explanation = ""
                    if not error:
                        explanation = get_natural_language_explanation(llm, user_question, pandas_code, result)
                    display_results(user_question, pandas_code, result, explanation, error)
        else:
            st.info("Please upload a CSV file to get started.")
    
    # Document Q&A Tab
    with tab2:
        st.header("Document Q&A")
        input_type = st.selectbox("Select input type", ["Upload PDF", "Enter URL"])
        
        if input_type == "Upload PDF":
            uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])
            if uploaded_pdf:
                current_input = {"type": "pdf", "value": uploaded_pdf.name}
                if 'current_input' not in st.session_state or st.session_state.current_input != current_input:
                    with st.spinner("Processing PDF..."):
                        try:
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                                tmp.write(uploaded_pdf.getvalue())
                                loader = PyPDFLoader(tmp.name)
                                documents = loader.load()
                            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                            splits = text_splitter.split_documents(documents)
                            embeddings = OpenAIEmbeddings()
                            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
                            retriever = vectorstore.as_retriever()
                            qa_chain = RetrievalQA.from_chain_type(
                                llm=llm,
                                chain_type="stuff",
                                retriever=retriever
                            )
                            st.session_state.current_input = current_input
                            st.session_state.vectorstore = vectorstore
                            st.session_state.qa_chain = qa_chain
                            st.success("PDF loaded and processed successfully.")
                        except Exception as e:
                            st.error(f"Error processing PDF: {e}")
        
        elif input_type == "Enter URL":
            url = st.text_input("Enter the URL")
            if url:
                current_input = {"type": "url", "value": url}
                if 'current_input' not in st.session_state or st.session_state.current_input != current_input:
                    with st.spinner("Processing URL..."):
                        try:
                            loader = WebBaseLoader(url)
                            documents = loader.load()
                            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                            splits = text_splitter.split_documents(documents)
                            embeddings = OpenAIEmbeddings()
                            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
                            retriever = vectorstore.as_retriever()
                            qa_chain = RetrievalQA.from_chain_type(
                                llm=llm,
                                chain_type="stuff",
                                retriever=retriever
                            )
                            st.session_state.current_input = current_input
                            st.session_state.vectorstore = vectorstore
                            st.session_state.qa_chain = qa_chain
                            st.success("URL loaded and processed successfully.")
                        except Exception as e:
                            st.error(f"Error processing URL: {e}")
        
        # Question input for documents
        if 'qa_chain' in st.session_state:
            question = st.text_area("Ask a question about the document")
            if st.button("Get Answer"):
                with st.spinner("Generating answer..."):
                    try:
                        answer = st.session_state.qa_chain.run(question)
                        st.markdown("### Answer")
                        st.write(answer)
                    except Exception as e:
                        st.error(f"Error generating answer: {e}")

if __name__ == "__main__":
    main()