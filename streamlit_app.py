"""
Streamlit demo app demonstrating code merge issue in Databricks Agent
and an improved "smart merge patch" approach.

This app shows:
1. The problem: LLM suggestions replace entire cell content
2. The solution: Smart merge that patches code intelligently
"""

import os
import re
import streamlit as st
import google.generativeai as genai
from difflib import unified_diff

# Configuration
# Available Gemini models (user can select in sidebar)
# Note: Some models may not be available depending on your API key/region
AVAILABLE_MODELS = [
    "gemini-pro",           # Most widely available
    "gemini-1.5-pro",       # Latest Pro model
    "gemini-1.5-flash"      # Faster model (may not be available in all regions)
]
DEFAULT_MODEL = "gemini-pro"

# Original Spark code cell
ORIGINAL_CODE = '''spark.sql("USE dq_demo")

car_csv_path = "/Volumes/workspace/dq_demo/car_sales_data/car_sales_data.csv"

car_sales_df = (
    spark.read
         .option("header", "true")
         .option("inferSchema", "true")
         .csv(car_csv_path)
)

from pyspark.sql.functions import col

car_sales_df.write.mode("overwrite").saveAsTable("silver_car_sales")

print("✅ Created table dq_demo.silver_car_sales from", car_csv_path)

print("Schema for car_sales_data.csv:")
car_sales_df.printSchema()

display(car_sales_df)

car_sales_df.write.mode("overwrite").saveAsTable("silver_car_sales")

print("✅ Created table dq_demo.silver_car_sales from", car_csv_path)
'''

# Error message
ERROR_OUTPUT = '''AnalysisException: [_LEGACY_ERROR_TEMP_DELTA_0007] A schema mismatch detected when writing to the Delta table `dq_demo`.`silver_car_sales`. 
Cannot safely cast `Engine size`: string to `Engine_size`: string.
Cannot safely cast `Fuel type`: string to `Fuel_type`: string.
Cannot safely cast `Year of manufacture`: string to `Year_of_manufacture`: string.

To enable schema migration, please set:
'.option("mergeSchema", "true")'
or
'.option("overwriteSchema", "true")'
'''


def call_gemini(prompt: str, api_key: str = None, model_name: str = None) -> str:
    """
    Call Gemini API with the given prompt.
    
    Args:
        prompt: The prompt to send to Gemini
        api_key: Optional API key (if None, will check session state and env var)
        model_name: Optional model name (if None, will use session state or default)
        
    Returns:
        The response text from Gemini
        
    Raises:
        ValueError: If API key is not set or invalid
    """
    # Get API key from parameter, session state, or environment variable
    if not api_key:
        api_key = st.session_state.get('gemini_api_key') or os.getenv("GEMINI_API_KEY")
    
    # Clean the API key - remove whitespace
    if api_key:
        api_key = api_key.strip()
    
    if not api_key:
        raise ValueError("Gemini API key is not set. Please enter it in the sidebar.")
    
    # Basic validation - Gemini API keys typically start with "AIza"
    if not api_key.startswith("AIza"):
        raise ValueError(
            "API key format looks incorrect. Gemini API keys typically start with 'AIza'. "
            "Please check your API key and make sure you copied it completely."
        )
    
    # Get model name from parameter, session state, or default
    if not model_name:
        model_name = st.session_state.get('gemini_model', DEFAULT_MODEL)
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        error_msg = str(e)
        if "API_KEY_INVALID" in error_msg or "API key not valid" in error_msg:
            return (
                f"❌ Invalid API Key Error: {error_msg}\n\n"
                "Please check:\n"
                "1. Make sure you copied the entire API key (they're usually long)\n"
                "2. Verify there are no extra spaces before/after the key\n"
                "3. Get a new API key from: https://makersuite.google.com/app/apikey\n"
                "4. Make sure the API key hasn't been revoked or expired"
            )
        elif "404" in error_msg and "models" in error_msg.lower():
            return (
                f"❌ Model Not Found Error: {error_msg}\n\n"
                f"The model '{model_name}' is not available with your API key.\n\n"
                "**Solution:** Try switching to 'gemini-pro' in the sidebar model selector.\n"
                "Some models may not be available in all regions or API versions."
            )
        return f"Error calling Gemini API: {error_msg}"


def create_inline_diff_view(original_code: str, merged_code: str) -> str:
    """
    Create an inline diff view (like Cursor AI) showing code with green highlighting for additions.
    Shows the final merged code with visual indicators of what was added.
    Only highlights NEW lines that were added, not original code.
    """
    original_lines = original_code.split('\n')
    merged_lines = merged_code.split('\n')
    
    # Use SequenceMatcher or unified_diff to properly track additions
    from difflib import SequenceMatcher
    
    matcher = SequenceMatcher(None, original_lines, merged_lines)
    
    # Track which lines in merged code are NEW additions
    added_line_indices = set()
    
    # Process matching blocks to find additions
    merged_idx = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            # Lines match - skip them (not added)
            merged_idx += (j2 - j1)
        elif tag == 'replace':
            # Original lines replaced with new lines - mark new lines as added
            for j in range(j1, j2):
                added_line_indices.add(merged_idx)
                merged_idx += 1
        elif tag == 'delete':
            # Lines deleted from original - don't increment merged index
            pass
        elif tag == 'insert':
            # New lines inserted - mark as added
            for j in range(j1, j2):
                added_line_indices.add(merged_idx)
                merged_idx += 1
    
    # Build HTML showing merged code with highlighting
    html = '<div style="font-family: \'Monaco\', \'Menlo\', monospace; font-size: 13px; line-height: 1.6; background: #fff; border: 1px solid #d1d9e0; border-radius: 6px; padding: 12px; overflow-x: auto; max-height: 500px; overflow-y: auto;">'
    
    for i, line in enumerate(merged_lines):
        escaped = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        
        if i in added_line_indices:
            # Green background for added lines (like Cursor AI)
            html += f'<div style="background-color: #e6ffed; padding: 4px 8px; margin: 2px 0; border-left: 4px solid #28a745; white-space: pre;"><span style="color: #28a745; font-weight: bold; margin-right: 6px;">+</span><span style="color: #24292e;">{escaped}</span></div>'
        else:
            # Normal line (unchanged)
            html += f'<div style="padding: 4px 8px; margin: 2px 0; white-space: pre; color: #24292e;">{escaped}</div>'
    
    html += '</div>'
    return html


def create_diff_view(original_code: str, merged_code: str) -> str:
    """
    Create a GitHub-style diff view showing what changed.
    Returns HTML with + and - highlighting.
    """
    original_lines = original_code.split('\n')
    merged_lines = merged_code.split('\n')
    
    diff_html = '<div style="font-family: \'Monaco\', \'Menlo\', \'Ubuntu Mono\', monospace; font-size: 13px; line-height: 1.6; background: #f6f8fa; border: 1px solid #d1d9e0; border-radius: 6px; padding: 16px; overflow-x: auto;">'
    
    # Use unified_diff to get the changes
    diff = list(unified_diff(original_lines, merged_lines, lineterm='', n=5))
    
    line_num = 0
    for line in diff:
        if line.startswith('+++') or line.startswith('---'):
            continue
        elif line.startswith('@@'):
            # Show context line numbers
            diff_html += f'<div style="background-color: #f1f8ff; padding: 4px 8px; margin: 8px 0; color: #586069; font-weight: bold; border-left: 3px solid #0366d6;">{line}</div>'
        elif line.startswith('+'):
            # Added line (green background)
            line_num += 1
            escaped = line[1:].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            diff_html += f'<div style="background-color: #e6ffed; padding: 2px 12px; margin: 1px 0; border-left: 3px solid #28a745; white-space: pre;"><span style="color: #28a745; font-weight: bold; margin-right: 8px;">+</span><span style="color: #24292e;">{escaped}</span></div>'
        elif line.startswith('-'):
            # Removed line (red background)
            escaped = line[1:].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            diff_html += f'<div style="background-color: #ffeef0; padding: 2px 12px; margin: 1px 0; border-left: 3px solid #d73a49; white-space: pre;"><span style="color: #d73a49; font-weight: bold; margin-right: 8px;">-</span><span style="color: #24292e;">{escaped}</span></div>'
        elif line.startswith(' '):
            # Context line (no highlighting)
            line_num += 1
            escaped = line[1:].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            diff_html += f'<div style="padding: 2px 12px; margin: 1px 0; color: #24292e; white-space: pre;"><span style="color: #999; margin-right: 8px;"> </span>{escaped}</div>'
    
    diff_html += '</div>'
    return diff_html


def get_diff_summary(original_code: str, merged_code: str) -> tuple:
    """
    Get summary of changes: lines added, removed, unchanged.
    """
    original_lines = original_code.split('\n')
    merged_lines = merged_code.split('\n')
    
    diff = list(unified_diff(original_lines, merged_lines, lineterm='', n=0))
    
    added = sum(1 for line in diff if line.startswith('+') and not line.startswith('+++'))
    removed = sum(1 for line in diff if line.startswith('-') and not line.startswith('---'))
    
    return added, removed, len(original_lines) - removed


def smart_merge_patch(original_code: str, llm_snippet: str) -> str:
    """
    Intelligently merge an LLM code snippet into the original code.
    
    Strategy:
    1. Clean snippet - remove duplicate code that already exists in original
    2. Check if LLM snippet includes a write statement
    3. If yes: Replace the first write statement (snippet already has write)
    4. If no: Insert snippet before first write statement
    5. Keep all other code intact
    
    Args:
        original_code: The original code cell content
        llm_snippet: The code snippet suggested by the LLM
        
    Returns:
        The merged code with the snippet inserted/replaced at the appropriate location
    """
    lines = original_code.split('\n')
    original_lower = original_code.lower()
    
    # Find the first occurrence of 'car_sales_df.write' (where error occurs)
    write_index = -1
    for i in range(len(lines)):
        if 'car_sales_df.write' in lines[i]:
            write_index = i
            break
    
    if write_index == -1:
        # If we can't find the write statement, just prepend the snippet
        return llm_snippet + '\n\n' + original_code
    
    # Clean the snippet - extract only the unique fix code
    # Strategy: Find the actual fix (select/withColumnRenamed) and keep from there
    snippet_lines = llm_snippet.strip().split('\n')
    cleaned_snippet_lines = []
    
    # Find where the actual fix starts (select, withColumnRenamed, or alias)
    fix_start_index = -1
    for i, line in enumerate(snippet_lines):
        line_lower = line.strip().lower()
        if ('select' in line_lower or 'withcolumnrenamed' in line_lower or 
            'alias' in line_lower or 'car_sales_df.write' in line_lower):
            fix_start_index = i
            break
    
    if fix_start_index >= 0:
        # Keep from the fix onwards, but check if there's a comment before it
        # Look backwards for a comment
        comment_index = fix_start_index
        for i in range(fix_start_index - 1, -1, -1):
            if snippet_lines[i].strip().startswith('#'):
                comment_index = i
                break
        
        # Keep comment + fix code
        cleaned_snippet_lines = snippet_lines[comment_index:]
    else:
        # No clear fix pattern found, use original snippet
        cleaned_snippet_lines = snippet_lines
    
    # Remove duplicate import statements
    cleaned_snippet_lines = [line for line in cleaned_snippet_lines 
                             if not (line.strip().startswith('from ') and line.strip() in original_code)]
    
    # Remove duplicate DataFrame creation blocks
    # If snippet starts with car_sales_df = (spark.read, it's likely a duplicate
    if cleaned_snippet_lines and 'car_sales_df = (' in cleaned_snippet_lines[0]:
        snippet_text = '\n'.join(cleaned_snippet_lines[:5]).lower()
        if 'spark.read' in snippet_text and '.option("header"' in snippet_text:
            # This is a duplicate, find where the actual fix starts
            for i, line in enumerate(cleaned_snippet_lines):
                line_lower = line.strip().lower()
                if ('select' in line_lower or 'withcolumnrenamed' in line_lower or 
                    'alias' in line_lower or 'car_sales_df.write' in line_lower):
                    # Keep from here
                    cleaned_snippet_lines = cleaned_snippet_lines[i:]
                    break
    
    # If we cleaned everything, use original snippet (fallback)
    if not cleaned_snippet_lines:
        cleaned_snippet_lines = snippet_lines
    
    snippet_has_write = any('car_sales_df.write' in line for line in cleaned_snippet_lines)
    
    # Add comment if snippet doesn't have one
    if cleaned_snippet_lines and not cleaned_snippet_lines[0].strip().startswith('#'):
        cleaned_snippet_lines = ['# Rename columns from spaced names to snake_case-like names'] + cleaned_snippet_lines
    
    if snippet_has_write:
        # Snippet includes write statement - REPLACE the first write
        # Find where the write statement ends (handle single-line and multi-line)
        write_end_index = write_index + 1
        
        # Check if it's a multi-line statement (next line continues with .mode, etc.)
        if write_index < len(lines) - 1:
            next_line = lines[write_index + 1].strip()
            if next_line and (next_line.startswith('.') or next_line.startswith('(')):
                # Multi-line statement - find where it ends
                for j in range(write_index + 1, len(lines)):
                    line_stripped = lines[j].strip()
                    if not line_stripped:
                        continue
                    if not (line_stripped.startswith('.') or line_stripped.startswith('(') or line_stripped.startswith(')')):
                        write_end_index = j
                        break
                else:
                    write_end_index = len(lines)
        
        # Replace the write statement with the cleaned snippet
        result_lines = (
            lines[:write_index] +  # All code before the write statement
            cleaned_snippet_lines +  # LLM suggestion (includes write, no duplicates)
            [''] +                 # Add blank line for readability
            lines[write_end_index:] # All code after the write statement
        )
    else:
        # Snippet doesn't include write - INSERT before first write
        result_lines = (
            lines[:write_index] +  # All code before the write statement
            cleaned_snippet_lines +  # LLM suggestion (column renaming, no duplicates)
            [''] +                 # Add blank line for readability
            lines[write_index:]    # Write statement and all code after
        )
    
    return '\n'.join(result_lines)


def build_gemini_prompt(code: str, error: str) -> str:
    """
    Build the prompt to send to Gemini for code fixing.
    
    Args:
        code: The current code cell content
        error: The error message from Spark
        
    Returns:
        The formatted prompt string
    """
    return f"""Analyze this Spark code and error.

Code:
```python
{code}
```

Error:
```
{error}
```

Suggest only the minimal code change needed to fix the schema mismatch by renaming columns from spaced names to underscored names.

Output a Python snippet that starts with `car_sales_df = (` and ends with `saveAsTable("silver_car_sales")`.

Do not include any explanations or markdown formatting - just the raw Python code snippet."""


# Streamlit UI
st.set_page_config(
    page_title="Databricks Agent Code Merge Demo",
    page_icon="🔧",
    layout="wide"
)

st.title("🔧 Databricks Agent Code Merge Demo")
st.markdown("""
This demo reproduces a code merge issue seen in Databricks Agent during a hackathon.
It shows the problem (entire cell replacement) and demonstrates an improved "smart merge patch" approach.
""")

# Initialize session state
if 'current_code' not in st.session_state:
    st.session_state.current_code = ORIGINAL_CODE
if 'original_code_snapshot' not in st.session_state:
    st.session_state.original_code_snapshot = ORIGINAL_CODE
if 'llm_suggestion' not in st.session_state:
    st.session_state.llm_suggestion = ""
if 'show_diff' not in st.session_state:
    st.session_state.show_diff = False
if 'code_updated' not in st.session_state:
    st.session_state.code_updated = False
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = ""
if 'gemini_model' not in st.session_state:
    st.session_state.gemini_model = DEFAULT_MODEL

# Sidebar for API key configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("---")
    api_key_input = st.text_input(
        "Gemini API Key",
        value=st.session_state.gemini_api_key,
        type="password",
        help="Enter your Gemini API key. Get one from https://makersuite.google.com/app/apikey",
        placeholder="AIza... (starts with AIza)"
    )
    # Clean the API key input (trim whitespace)
    st.session_state.gemini_api_key = api_key_input.strip() if api_key_input else ""
    
    # Use found models if available, otherwise use default list
    models_to_show = AVAILABLE_MODELS
    if 'available_models_found' in st.session_state and st.session_state.available_models_found:
        models_to_show = sorted(st.session_state.available_models_found)
    
    current_model = st.session_state.gemini_model
    # If current model is not in available models, find best coding model
    if current_model not in models_to_show:
        # Recommend best model for coding
        recommended_model = None
        # Priority: 1.5-pro-latest > 1.5-pro > pro > flash
        for pattern in ['gemini-1.5-pro-latest', 'gemini-1.5-pro', 'gemini-pro', 'gemini-1.5-flash']:
            for model in models_to_show:
                if pattern in model.lower():
                    recommended_model = model
                    break
            if recommended_model:
                break
        # Fallback to first model if no match
        if not recommended_model and models_to_show:
            recommended_model = models_to_show[0]
        st.session_state.gemini_model = recommended_model or DEFAULT_MODEL
        current_model = st.session_state.gemini_model
    
    try:
        model_index = models_to_show.index(current_model) if current_model in models_to_show else 0
    except (ValueError, AttributeError):
        model_index = 0
        st.session_state.gemini_model = models_to_show[0] if models_to_show else DEFAULT_MODEL
    
    # Determine best model for coding recommendation
    best_coding_model = None
    for pattern in ['gemini-1.5-pro-latest', 'gemini-1.5-pro', 'gemini-pro']:
        for model in models_to_show:
            if pattern in model.lower():
                best_coding_model = model
                break
        if best_coding_model:
            break
    
    help_text = "Select which Gemini model to use."
    if best_coding_model:
        help_text += f" 💡 Recommended for coding: '{best_coding_model}'"
    
    model_selection = st.selectbox(
        "Gemini Model",
        options=models_to_show,
        index=model_index,
        help=help_text
    )
    st.session_state.gemini_model = model_selection
    
    # Show recommendation
    if best_coding_model and model_selection != best_coding_model:
        st.info(f"💡 **Tip:** For coding tasks, '{best_coding_model}' is recommended (better code quality).")
    
    if st.session_state.gemini_api_key:
        # Validate format
        if st.session_state.gemini_api_key.startswith("AIza"):
            st.success("✅ API key set")
        else:
            st.warning("⚠️ API key format looks incorrect (should start with 'AIza')")
    else:
        st.warning("⚠️ API key required to use LLM features")
    
    st.markdown("---")
    
    # Button to check available models
    if st.session_state.gemini_api_key and st.session_state.gemini_api_key.startswith("AIza"):
        if st.button("🔍 Check Available Models", use_container_width=True):
            with st.spinner("Checking available models..."):
                try:
                    genai.configure(api_key=st.session_state.gemini_api_key)
                    models = genai.list_models()
                    
                    available_models_found = []
                    for model in models:
                        if 'generateContent' in model.supported_generation_methods:
                            model_name = model.name.replace('models/', '')
                            available_models_found.append(model_name)
                    
                    if available_models_found:
                        st.success(f"✅ Found {len(available_models_found)} available model(s)")
                        
                        # Store found models in session state
                        st.session_state.available_models_found = available_models_found
                        
                        # Find best model for coding
                        best_for_coding = None
                        for pattern in ['gemini-1.5-pro-latest', 'gemini-1.5-pro', 'gemini-pro']:
                            for model in available_models_found:
                                if pattern in model.lower():
                                    best_for_coding = model
                                    break
                            if best_for_coding:
                                break
                        
                        if best_for_coding:
                            st.success(f"🎯 **Best for coding:** `{best_for_coding}`")
                            # Auto-select best model if current one isn't available
                            if st.session_state.gemini_model not in available_models_found:
                                st.session_state.gemini_model = best_for_coding
                                st.info(f"✅ Auto-selected '{best_for_coding}' for you!")
                        
                        # Show all models in expandable section
                        with st.expander(f"📋 View all {len(available_models_found)} available models"):
                            for model in sorted(available_models_found):
                                is_best = " ⭐ BEST FOR CODING" if model == best_for_coding else ""
                                st.code(f"{model}{is_best}", language=None)
                        
                        st.info("💡 The model dropdown above has been updated with all available models.")
                    else:
                        st.warning("No models found with generateContent support.")
                except Exception as e:
                    st.error(f"Error checking models: {str(e)}")
        
        # Show found models if available
        if 'available_models_found' in st.session_state and st.session_state.available_models_found:
            st.caption(f"📋 {len(st.session_state.available_models_found)} model(s) available")
    
    st.markdown("---")
    with st.expander("ℹ️ How to get API key"):
        st.markdown("""
        1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
        2. Sign in with your Google account
        3. Click "Create API Key" or "Get API Key"
        4. Copy the entire key (it starts with `AIza` and is quite long)
        5. Paste it in the field above
        
        **Note:** Make sure to copy the entire key - they're usually 39+ characters long.
        """)
    
    st.caption("Enter your API key above to enable the 'Ask LLM for fix' button.")

# Create two columns layout (like Databricks)
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("📝 Current cell code")
    
    # Store original code snapshot when LLM suggestion is first generated
    if st.session_state.llm_suggestion and not st.session_state.show_diff:
        st.session_state.original_code_snapshot = st.session_state.current_code
    
    # Show inline diff view if suggestion exists (like Cursor AI)
    if st.session_state.llm_suggestion:
        # Calculate merged code for preview
        merged_preview = smart_merge_patch(
            st.session_state.original_code_snapshot,
            st.session_state.llm_suggestion
        )
        
        # Show inline diff in code editor (green for additions, red for deletions)
        st.markdown("**📊 Inline Diff View** (Green = Added, Red = Removed)")
        diff_html = create_inline_diff_view(st.session_state.original_code_snapshot, merged_preview)
        st.markdown(diff_html, unsafe_allow_html=True)
        
        # Also show editable code editor below
        st.markdown("---")
        st.markdown("**📝 Editable Code:**")
        # Always use the same key so Streamlit preserves state correctly
        code_editor = st.text_area(
            "Current cell code",
            value=st.session_state.current_code,
            height=300,
            key="code_editor",
            label_visibility="collapsed"
        )
        # Update session state with editor value
        st.session_state.current_code = code_editor
    else:
        # Normal code editor when no suggestion
        # If code was just updated by merge, use a different key to force refresh
        editor_key = "code_editor_updated" if st.session_state.get("code_updated", False) else "code_editor"
        if st.session_state.get("code_updated", False):
            st.session_state.code_updated = False  # Reset flag
        
        code_editor = st.text_area(
            "Current cell code",
            value=st.session_state.current_code,
            height=400,
            key=editor_key,
            label_visibility="collapsed"
        )
        # Update session state with editor value
        st.session_state.current_code = code_editor
    
    st.markdown("---")
    st.subheader("⚠️ Error output")
    st.text_area(
        "Error output",
        value=ERROR_OUTPUT,
        height=150,
        disabled=True,
        label_visibility="collapsed"
    )
    
    # Ask LLM button
    if st.button("🤖 Ask LLM for fix", type="primary", use_container_width=True):
        if not st.session_state.gemini_api_key and not os.getenv("GEMINI_API_KEY"):
            st.error("⚠️ Please enter your Gemini API key in the sidebar first!")
        else:
            with st.spinner("Calling Gemini API..."):
                try:
                    prompt = build_gemini_prompt(
                        st.session_state.current_code,
                        ERROR_OUTPUT
                    )
                    response = call_gemini(prompt, st.session_state.gemini_api_key, st.session_state.gemini_model)
                    # Clean up the response - remove markdown code blocks if present
                    cleaned_response = response.strip()
                    # Remove markdown code fences if present
                    if cleaned_response.startswith("```"):
                        lines = cleaned_response.split('\n')
                        # Remove first line (```python or ```)
                        if len(lines) > 1:
                            cleaned_response = '\n'.join(lines[1:])
                        # Remove last line (```)
                        if cleaned_response.endswith("```"):
                            cleaned_response = cleaned_response[:-3].strip()
                    st.session_state.llm_suggestion = cleaned_response.strip()
                    st.session_state.original_code_snapshot = st.session_state.current_code
                    st.session_state.show_diff = True
                    st.success("✅ Got suggestion from Gemini!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.session_state.llm_suggestion = ""

with col2:
    st.subheader("💡 LLM Suggestion")
    
    if st.session_state.llm_suggestion:
        # Show the LLM suggestion
        st.text_area(
            "LLM suggestion",
            value=st.session_state.llm_suggestion,
            height=200,
            disabled=True,
            label_visibility="collapsed"
        )
        
        # Calculate merged preview
        merged_preview = smart_merge_patch(
            st.session_state.original_code_snapshot,
            st.session_state.llm_suggestion
        )
        
        # Show diff summary
        added, removed, unchanged = get_diff_summary(st.session_state.original_code_snapshot, merged_preview)
        st.info(f"📈 **{added}** lines will be added | **{removed}** lines removed | **{unchanged}** unchanged")
        
        st.markdown("---")
        st.markdown("### Apply Changes:")
        
        col_replace, col_merge = st.columns(2)
        
        with col_replace:
            if st.button("❌ Replace entire cell", use_container_width=True, help="Bad: Deletes all original code"):
                st.session_state.current_code = st.session_state.llm_suggestion
                st.session_state.llm_suggestion = ""
                st.session_state.show_diff = False
                st.success("⚠️ Code replaced (all original code deleted!)")
                st.rerun()
        
        with col_merge:
            if st.button("✅ Smart merge patch", type="primary", use_container_width=True, key="merge_button", help="Good: Preserves all code, inserts fix"):
                # Use the original snapshot for merge (not current editor which might have been edited)
                merged_result = smart_merge_patch(
                    st.session_state.original_code_snapshot,
                    st.session_state.llm_suggestion
                )
                # IMPORTANT: Update current_code BEFORE clearing suggestion
                # This ensures the merged code is set before rerun
                st.session_state.current_code = merged_result
                # Update snapshot to merged result  
                st.session_state.original_code_snapshot = merged_result
                # Clear suggestion and reset state
                st.session_state.llm_suggestion = ""
                st.session_state.show_diff = False
                # Add a flag to force text area update
                st.session_state.code_updated = True
                # Force rerun to update UI
                st.rerun()
        
        st.markdown("---")
        st.markdown("### 📋 What will change:")
        st.caption("Green highlights show where code will be added, red shows what will be removed.")
    else:
        st.info("👆 Click 'Ask LLM for fix' to get a code suggestion")
        st.markdown("---")
        st.markdown("""
        **How it works:**
        1. Enter your API key in sidebar
        2. Click "Ask LLM for fix"
        3. See the diff view (green = added, red = removed)
        4. Choose to merge or replace
        """)

# Footer
st.markdown("---")
st.caption("Demo app showing improved code merge behavior for AI coding assistants")

