import streamlit as st
import json
import _snowflake
import re
import pandas as pd
from snowflake.snowpark.context import get_active_session
from collections import Counter
import uuid

# Initialize Snowflake session
session = get_active_session()

# Cortex API Configuration
API_ENDPOINT = "/api/v2/cortex/agent:run"
API_TIMEOUT = 50000  # in milliseconds

# Single Semantic Model Configuration
SEMANTIC_MODEL = '@"AI"."DWH_MART"."GRANTS"/grantsyaml_27.yaml'
CORTEX_SEARCH_SERVICES = "AI.DWH_MART.GRANTS_SEARCH_SERVICES"

st.set_page_config(page_title="📄 Multi-Model Cortex Assistant", layout="wide")
st.title("📄 AI Assistant for GRANTS")

# Custom CSS to hide Streamlit branding
st.markdown("""
<style>
#MainMenu, header, footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Suggested questions
suggested_questions = [
    "What is the posted budget for awards 41001, 41002, 41003, 41005, 41007, and 41018 by date?",
    "Give me date wise award breakdowns",
    "Give me award breakdowns",
    "Give me date wise award budget, actual award posted,award encunbrance posted,award encumbrance approved",
    "What is the task actual posted by award name?",
    "What is the award budget posted by date for these awards?",
    "What is the total award encumbrance posted for these awards?",
    "What is the total amount of award encumbrances approved?",
    "What is the total actual award posted for these awards?",
    "what is the award budget posted?",
    "what is this document about",
    "Subjec areas",
    "explain five layers in High level Architecture"
]

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False
if 'current_query_to_process' not in st.session_state:
    st.session_state.current_query_to_process = None
if 'show_suggested_buttons' not in st.session_state:
    st.session_state.show_suggested_buttons = False

def run_snowflake_query(query):
    """Executes a SQL query against Snowflake and returns results as a Pandas DataFrame."""
    try:
        if not query:
            return None, "⚠️ No SQL query generated."
        df = session.sql(query)
        pandas_df = df.to_pandas()
        return pandas_df, None
    except Exception as e:
        return None, f"❌ SQL Execution Error: {str(e)}"

def is_structured_query(query: str):
    """
    Determines if a query is structured based on keywords typically associated with data queries.
    Enhanced to reduce false positives for unstructured queries.
    """
    structured_keywords = [
        "total", "show", "top", "funding", "net increase", "net decrease", "group by", "order by",
        "how much", "give", "count", "avg", "max", "min", "least", "highest", "by year",
        "how many", "total amount", "version", "scenario", "forecast", "year", "savings",
        "award", "position", "budget", "allocation", "expenditure", "department", "variance",
        "breakdown", "comparison", "change"
    ]
    unstructured_keywords = [
        "describe", "introduction", "summary", "tell me about", "overview", "explain"
    ]
    query_lower = query.lower()
    # Check for unstructured indicators first
    if any(keyword in query_lower for keyword in unstructured_keywords):
        structured_score = sum(1 for keyword in structured_keywords if keyword in query_lower)
        if structured_score < 2:  # Require at least 2 structured keywords to override
            return False
    return any(keyword in query_lower for keyword in structured_keywords)

def is_unstructured_query(query: str):
    unstructured_keywords = [
        "policy", "document", "description", "summary", "highlight", "explain", "describe", "guidelines",
        "procedure", "how to", "define", "definition", "rules", "steps", "overview",
        "objective", "purpose", "benefits", "importance", "impact", "details", "regulation",
        "requirement", "compliance", "when to", "where to", "meaning", "interpretation",
        "clarify", "note", "explanation", "instructions"
    ]
    query_lower = query.lower()
    return any(word in query_lower for word in unstructured_keywords)

def detect_yaml_or_sql_intent(query: str):
    """Detects if a query is asking for information about the semantic model (YAML) or SQL structure."""
    yaml_keywords = ["yaml", "semantic model", "metric", "dimension", "table", "column", "sql for", "structure of"]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in yaml_keywords)

def preprocess_query(query: str):
    """Extracts key terms from the query to improve search relevance."""
    query_lower = query.lower()
    # Split on whitespace
    tokens = query_lower.split()
    
    # Remove stopwords and non-informative words
    stopwords = set(['what', 'is', 'the', 'a', 'an', 'of', 'in', 'for', 'and', 'or', 'to'])
    key_terms = [token for token in tokens if token not in stopwords and token.isalnum()]
    
    # Simple normalization: remove common suffixes
    def normalize_term(term):
        return re.sub(r'(ing|s|ed)$', '', term)
    
    return [normalize_term(term) for term in key_terms]

def summarize_unstructured_answer(answer: str, query: str):
    """Summarizes unstructured text by ranking sentences based on query relevance with weighted scoring."""
    # Clean the answer
    answer = re.sub(r"^.*?Program\sOverview", "Program Overview", answer, flags=re.DOTALL)
    
    # Split on periods, question marks, or exclamation marks followed by whitespace
    sentences = re.split(r'(?<=\.|\?|\!)\s+', answer)
    sentences = [sent.strip() for sent in sentences if sent.strip()]
    
    if not sentences:
        return "No relevant content found."
    
    # Extract key terms from the query
    key_terms = preprocess_query(query)
    
    # Approximate inverse document frequency (IDF) for term weighting
    all_words = ' '.join(sentences).lower().split()
    word_counts = Counter(all_words)
    total_words = len(all_words) + 1  # Avoid division by zero
    term_weights = {term: max(1.0, 3.0 * (1 - word_counts.get(term.lower(), 0) / total_words)) for term in key_terms}
    
    # Score sentences based on weighted key terms
    scored_sentences = []
    for sent in sentences:
        sent_lower = sent.lower()
        score = sum(term_weights[term] for term in key_terms if term in sent_lower)
        scored_sentences.append((sent, score))
    
    # Sort sentences by score (descending) and select top 5
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    top_sentences = [sent for sent, score in scored_sentences[:5] if score > 0]
    
    if not top_sentences:
        # Fallback to first 5 sentences
        top_sentences = sentences[:5]
    
    return "\n\n".join(f"• {sent}" for sent in top_sentences)

def summarize(text: str, query: str):
    """Calls Snowflake Cortex SUMMARIZE function with cleaned input text, with local fallback."""
    try:
        # Clean text by removing excessive whitespace and special characters
        text = re.sub(r'\s+', ' ', text.strip())
        text = text.replace("'", "\\'")
        query_sql = f"SELECT SNOWFLAKE.CORTEX.SUMMARIZE('{text}') AS summary"
        result = session.sql(query_sql).collect()
        summary = result[0]["SUMMARY"]
        if summary and len(summary) > 50:  # Ensure summary is meaningful
            return summary
        raise Exception("Cortex SUMMARIZE returned empty or too short summary.")
    except Exception as e:
        if st.session_state.debug_mode:
            st.write(f"Debug: SUMMARIZE Function Error: {str(e)}. Using local fallback.")
        # Fallback: Use relevance scoring to select top sentences
        sentences = re.split(r'(?<=\.|\?|\!)\s+', text)
        sentences = [sent.strip() for sent in sentences if sent.strip() and len(sent) > 20]
        if not sentences:
            return "No relevant content found."
        
        key_terms = preprocess_query(query)
        all_words = ' '.join(sentences).lower().split()
        word_counts = Counter(all_words)
        total_words = len(all_words) + 1
        term_weights = {term: max(1.0, 3.0 * (1 - word_counts.get(term.lower(), 0) / total_words)) for term in key_terms}
        
        scored_sentences = []
        for sent in sentences:
            sent_lower = sent.lower()
            score = sum(term_weights[term] for term in key_terms if term in sent_lower)
            scored_sentences.append((sent, score))
        
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [sent for sent, score in scored_sentences[:3] if score > 0]
        if not top_sentences:
            top_sentences = sentences[:3]
        
        return "\n".join(top_sentences) if top_sentences else "No relevant content found."

def snowflake_api_call(query: str, is_structured: bool = False, selected_model=None, is_yaml=False):
    """
    Makes an API call to Snowflake Cortex, routing to text-to-SQL or search service
    based on the query type.
    """
    payload = {
        "model": "llama3.1-70b",
        "messages": [{"role": "user", "content": [{"type": "text", "text": query}]}],
        "tools": []
    }
    if is_structured or is_yaml:
        payload["tools"].append({"tool_spec": {"type": "cortex_analyst_text_to_sql", "name": "analyst1"}})
        payload["tool_resources"] = {"analyst1": {"semantic_model_file": selected_model}}
    else:
        payload["tools"].append({"tool_spec": {"type": "cortex_search", "name": "search1"}})
        payload["tool_resources"] = {"search1": {"name": CORTEX_SEARCH_SERVICES, "max_results": 10}}
    try:
        resp = _snowflake.send_snow_api_request("POST", API_ENDPOINT, {}, {}, payload, None, API_TIMEOUT)
        response = json.loads(resp["content"])
        if st.session_state.debug_mode:
            st.write(f"Debug: API Response for query '{query}': {response}")
        return response, None
    except Exception as e:
        return None, f"❌ API Request Failed: {str(e)}"

def process_sse_response(response, is_structured, query):
    """
    Processes the SSE response from Snowflake Cortex, extracting SQL/explanation
    for structured queries or search results for unstructured queries.
    """
    sql = ""
    explanation = ""
    search_results = []
    error = None
    if not response:
        return sql, explanation, search_results, "No response from API."
    try:
        for event in response:
            if isinstance(event, dict) and event.get('event') == "message.delta":
                data = event.get('data', {})
                delta = data.get('delta', {})
                for content_item in delta.get('content', []):
                    if content_item.get('type') == "tool_results":
                        tool_results = content_item.get('tool_results', {})
                        if 'content' in tool_results:
                            for result in tool_results['content']:
                                if result.get('type') == 'json':
                                    result_data = result.get('json', {})
                                    if is_structured:
                                        if 'sql' in result_data:
                                            sql = result_data.get('sql', '')
                                        if 'explanation' in result_data:
                                            explanation = result_data.get('explanation', '')
                                    else:
                                        if 'searchResults' in result_data:
                                            # Rank search results by relevance to query
                                            key_terms = preprocess_query(query)
                                            ranked_results = []
                                            for sr in result_data['searchResults']:
                                                text = sr["text"]
                                                text_lower = text.lower()
                                                score = sum(1 for term in key_terms if term in text_lower)
                                                ranked_results.append((text, score))
                                            ranked_results.sort(key=lambda x: x[1], reverse=True)
                                            # Process all results without strict deduplication
                                            search_results = [
                                                summarize_unstructured_answer(text, query)
                                                for text, _ in ranked_results
                                            ]
                                            # Filter out empty or irrelevant results
                                            search_results = [sr for sr in search_results if sr and "No relevant content found" not in sr]
        if not is_structured and not search_results:
            error = "No relevant search results returned from the search service."
    except Exception as e:
        error = f"❌ Error Processing Response: {str(e)}"
    if st.session_state.debug_mode:
        st.write(f"Debug: Processed Response - SQL: {sql}, Explanation: {explanation}, Search Results: {search_results}, Error: {error}")
    return sql.strip(), explanation.strip(), search_results, error

def format_results_for_history(df):
    """Formats a Pandas DataFrame into a Markdown table for chat history."""
    if df is None or df.empty:
        return "No data found."
    if len(df.columns) == 1:
        return str(df.iloc[0, 0])
    return df.to_markdown(index=False)

def process_query_and_display(query: str):
    """
    Processes a user query, interacts with Cortex, displays results,
    and updates session state.
    """
    # Reset the flag to hide suggested buttons at the start of a new query
    st.session_state.show_suggested_buttons = False

    # Add the user's query to the chat history and display it
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    response_content_for_history = ""

    with st.chat_message("assistant"):
        # Check for greeting inputs
        query_lower = query.lower().strip()
        if query_lower in ["hi", "hello"]:
            greeting_response = (
                "Hello! Welcome to the GRANTS AI Assistant! I'm here to help you explore and analyze "
                "grant-related data, answer questions about awards, budgets, and more, or provide insights "
                "from documents.\n\nHere are some questions you can try:\n"
                "- What is the posted budget for awards 41001, 41002, 41003, 41005, 41007, and 41018 by date?\n"
                "- Give me date-wise award breakdowns.\n"
                "- What is this document about?\n"
                "- List all subject areas.\n\n"
                "Feel free to ask anything, or pick one of the suggested questions to get started!"
            )
            st.markdown(greeting_response)
            response_content_for_history = greeting_response
            st.session_state.messages.append({"role": "assistant", "content": response_content_for_history})
            return  # Exit early after handling greeting

        with st.spinner("Thinking... 🤖"):
            is_structured = is_structured_query(query)
            is_yaml = detect_yaml_or_sql_intent(query)

            if is_structured or is_yaml:
                response_json, api_error = snowflake_api_call(query, is_structured=True, selected_model=SEMANTIC_MODEL, is_yaml=is_yaml)
                if api_error:
                    st.error(api_error)
                    response_content_for_history = api_error
                    st.session_state.show_suggested_buttons = True
                else:
                    final_sql, explanation, _, sse_error = process_sse_response(response_json, is_structured=True, query=query)
                    if sse_error:
                        st.error(sse_error)
                        response_content_for_history = sse_error
                        st.session_state.show_suggested_buttons = True
                    elif final_sql:
                        st.markdown("**📜 SQL Query:**")
                        st.code(final_sql, language='sql')
                        response_content_for_history += f"**📜 SQL Query:**\n```sql\n{final_sql}\n```\n"

                        if explanation:
                            st.markdown("**📘 Explanation:**")
                            st.write(explanation)
                            response_content_for_history += f"**📘 Explanation:**\n{explanation}\n"

                        results_df, query_error = run_snowflake_query(final_sql)
                        if query_error:
                            st.error(query_error)
                            response_content_for_history += query_error
                            st.session_state.show_suggested_buttons = True
                        elif results_df is not None and not results_df.empty:
                            st.markdown("**📊 Results:**")
                            if len(results_df.columns) == 1:
                                st.write(f"**{results_df.iloc[0, 0]}**")
                            else:
                                st.dataframe(results_df)
                            response_content_for_history += "**📊 Results:**\n" + format_results_for_history(results_df)
                        else:
                            st.markdown("⚠️ No data found for the generated SQL query.")
                            response_content_for_history += "⚠️ No data found for the generated SQL query.\n"
                            st.session_state.show_suggested_buttons = True
                    else:
                        st.markdown("⚠️ No SQL generated. Could not understand the structured/YAML query.")
                        response_content_for_history = "⚠️ No SQL generated. Could not understand the structured/YAML query.\n"
                        st.session_state.show_suggested_buttons = True
            else:
                response_json, api_error = snowflake_api_call(query, is_structured=False)
                if api_error:
                    st.error(api_error)
                    response_content_for_history = api_error
                    st.session_state.show_suggested_buttons = True
                else:
                    _, _, search_results, sse_error = process_sse_response(response_json, is_structured=False, query=query)
                    if sse_error:
                        st.error(sse_error)
                        response_content_for_history = sse_error
                        st.session_state.show_suggested_buttons = True
                    elif search_results:
                        st.markdown("**🔍 Document Highlights:**")
                        # Combine and summarize multiple search results
                        combined_results = "\n\n".join(search_results)
                        summarized_result = summarize(combined_results, query)
                        st.write(summarized_result)
                        response_content_for_history += f"**🔍 Document Highlights:**\n{summarized_result}\n"
                    else:
                        st.markdown(f"### I couldn't find information for: '{query}'")
                        st.markdown("Try rephrasing your question or selecting from the suggested questions below.")
                        response_content_for_history = f"### I couldn't find information for: '{query}'\nTry rephrasing your question or selecting from the suggested questions."
                        st.session_state.show_suggested_buttons = True

            st.session_state.messages.append({"role": "assistant", "content": response_content_for_history})

def main():
    # Sidebar setup
    st.sidebar.header("🔍 Ask About GRANTS Analytics")
    st.sidebar.info(f"📂 Current Model: **GRANTS**")
    st.session_state.debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=st.session_state.debug_mode)

    if st.sidebar.button("Refresh", key="new_conversation_button"):
        st.session_state.messages = []
        st.session_state.current_query_to_process = None
        st.session_state.show_suggested_buttons = False
        st.rerun()

    with st.sidebar:
        st.markdown("### 💡 Suggested Questions")
        for q in suggested_questions:
            if st.button(q, key=f"sidebar_suggested_{hash(q)}"):
                st.session_state.current_query_to_process = q
                st.rerun()
    # Display welcome message if no queries have been made
    if not st.session_state.messages:
        st.markdown("💡 **Welcome! I’m the Snowflake AI Assistant, ready to assist you with grant data analysis, summaries, and answers — simply type your question to get started**")
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    # Handle user input or suggested question clicks
    chat_input_query = st.chat_input("Ask a question...")
    if chat_input_query:
        st.session_state.current_query_to_process = chat_input_query

    if st.session_state.current_query_to_process:
        query_to_process = st.session_state.current_query_to_process
        st.session_state.current_query_to_process = None
        process_query_and_display(query_to_process)
        st.rerun()

    # Display suggested questions in the chat area if the query fails
    if st.session_state.show_suggested_buttons:
        st.markdown("---")
        st.markdown("### 💡 Try one of these questions:")
        cols = st.columns(2)
        for idx, q in enumerate(suggested_questions):
            with cols[idx % 2]:
                if st.button(q, key=f"chat_suggested_button_{hash(q)}"):
                    st.session_state.current_query_to_process = q
                    st.rerun()

if __name__ == "__main__":
    main()