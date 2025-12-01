# Databricks Agent Code Merge Demo

This Streamlit demo app reproduces a code merge issue seen in Databricks Agent during a hackathon and demonstrates an improved "smart merge patch" approach.

## 📖 Full Demo Explanation

See **[DEMO_EXPLANATION.md](DEMO_EXPLANATION.md)** for a complete explanation of the issue, root cause analysis, and demo script for presenting to stakeholders.

## Problem

**Context**: CSV file has column names with spaces (`"Engine size"`, `"Fuel type"`), but Delta table uses underscores (`"Engine_size"`, `"Fuel_type"`). This causes a schema mismatch error.

**The Issue**: When Databricks Agent suggests a code fix, clicking **"Replace active cell content"** replaces the **ENTIRE cell** with only the LLM snippet, **deleting all original code**:
- Database setup (`spark.sql("USE dq_demo")`)
- Variable definitions (`car_csv_path`)
- Import statements
- Print statements for debugging
- Schema printing
- Display calls

**What's Missing in RAG Function**: The RAG function treats LLM output as a **complete replacement** instead of a **patch/diff** that should be intelligently merged.

## Solution

This demo shows a "Smart merge patch" approach that intelligently merges the LLM suggestion into the existing code, preserving all original functionality.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your Gemini API key:
```bash
export GEMINI_API_KEY="your_api_key_here"
```

Get your free API key from [Google AI Studio](https://makersuite.google.com/app/apikey).

## Running the App

```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`.

## How It Works

1. **Current cell code**: Pre-filled with the original Spark code that causes a schema mismatch error
2. **Error output**: Shows the Delta AnalysisException error
3. **Ask LLM for fix**: Calls Gemini API to get a code suggestion
4. **Replace entire cell**: Demonstrates the problematic behavior (replaces all code)
5. **Smart merge patch**: Shows the improved behavior (merges snippet intelligently)

The smart merge strategy:
- Finds the last occurrence of `car_sales_df.write` in the code
- Inserts the LLM snippet before that line
- Preserves all original code (print statements, display, etc.)

## Demo Videos

> 🔗 **Demo 1 – Current Databricks behavior**  
> Demonstrates how “Replace active cell content” wipes out the entire cell even though the LLM snippet assumes the surrounding code still exists.  
> [Watch Demo 1](https://drive.google.com/file/d/1Ow4xnHDptOyFpPWlxPZZ5bWqLf4r3A15/view)

> 🔗 **Demo 2 – Smart merge prototype**  
> Streamlit proof-of-concept that asks an LLM for the same fix but applies it as a non-destructive patch, preserving the original code and inserting the snippet before the write operation.  
> [Watch Demo 2](https://drive.google.com/file/d/1HDd1SOygIFJCBWAWf4hr5Er1Lfbq9Lg4/view?usp=sharing)

## Files

- `streamlit_app.py`: Main Streamlit application
- `requirements.txt`: Python dependencies
