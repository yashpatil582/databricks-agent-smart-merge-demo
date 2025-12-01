# Databricks Agent Code Merge Issue - Demo Explanation

## Problem Statement

### Context
- **CSV File**: Has column names with spaces (e.g., `"Engine size"`, `"Fuel type"`, `"Year of manufacture"`)
- **Delta Table**: Already exists with underscores (e.g., `"Engine_size"`, `"Fuel_type"`, `"Year_of_manufacture"`)
- **Error**: Schema mismatch when writing DataFrame to Delta table

### The Issue: Complete Cell Replacement

When Databricks Agent suggests a code fix, clicking the **"Replace active cell content"** button replaces the **ENTIRE cell** with only the suggested snippet, **deleting all original code**.

#### Original Code (Before Fix)
```python
spark.sql("USE dq_demo")

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
```

#### What Databricks Agent Suggests (Partial Fix)
```python
car_sales_df = (
    car_sales_df
    .withColumnRenamed("Engine size", "Engine_size")
    .withColumnRenamed("Fuel type", "Fuel_type")
    .withColumnRenamed("Year of manufacture", "Year_of_manufacture")
)

car_sales_df.write.mode("overwrite").saveAsTable("silver_car_sales")
```

#### What Happens After "Replace" Button (❌ WRONG)
The entire cell is replaced with ONLY the snippet above, **losing**:
- `spark.sql("USE dq_demo")` - database context
- `car_csv_path` variable definition
- CSV reading logic
- Import statements
- Print statements for debugging
- Schema printing
- Display calls
- Second write statement (if needed)

#### What Should Happen (✅ CORRECT - Smart Merge)
```python
spark.sql("USE dq_demo")

car_csv_path = "/Volumes/workspace/dq_demo/car_sales_data/car_sales_data.csv"

car_sales_df = (
    spark.read
         .option("header", "true")
         .option("inferSchema", "true")
         .csv(car_csv_path)
)

from pyspark.sql.functions import col

# Rename columns from spaced names to snake_case-like names
car_sales_df = (
    car_sales_df
    .withColumnRenamed("Engine size", "Engine_size")
    .withColumnRenamed("Fuel type", "Fuel_type")
    .withColumnRenamed("Year of manufacture", "Year_of_manufacture")
)

car_sales_df.write.mode("overwrite").saveAsTable("silver_car_sales")

print("✅ Created table dq_demo.silver_car_sales from", car_csv_path)

print("Schema for car_sales_data.csv:")
car_sales_df.printSchema()

display(car_sales_df)

car_sales_df.write.mode("overwrite").saveAsTable("silver_car_sales")

print("✅ Created table dq_demo.silver_car_sales from", car_csv_path)
```

**Key Difference**: The column renaming code is **inserted** before the write statement, preserving ALL original code.

---

## Root Cause: What's Missing in RAG Function

### Current Behavior (Problematic)
The RAG function in Databricks Agent:
1. ✅ Correctly identifies the issue (schema mismatch)
2. ✅ Generates the correct fix (column renaming)
3. ❌ **Treats the LLM output as a complete replacement** instead of a **patch/diff**
4. ❌ **Doesn't preserve context** (imports, setup code, debugging statements)

### What Should Happen (Solution)
The RAG function should:
1. Identify the issue ✅
2. Generate the fix ✅
3. **Treat LLM output as a patch/diff** (not full replacement)
4. **Merge intelligently** by:
   - Finding the insertion point (e.g., before `car_sales_df.write`)
   - Preserving all original code
   - Inserting only the new code snippet
   - Maintaining code structure and flow

---

## Demo Flow

### Step 1: Show the Problem
1. Display original code with schema mismatch error
2. Click "Ask LLM for fix" → Get suggestion
3. Click **"Replace entire cell"** → Show how it deletes everything
4. **Result**: Only snippet remains, all context lost ❌

### Step 2: Show the Solution
1. Reset to original code
2. Click "Ask LLM for fix" → Get same suggestion
3. Click **"Smart merge patch"** → Show intelligent merge
4. **Result**: All code preserved + fix inserted at correct location ✅

### Step 3: Explain the Difference
- **Bad**: LLM output treated as **complete replacement**
- **Good**: LLM output treated as **patch/diff** with intelligent merge

---

## Technical Implementation: Smart Merge Strategy

The `smart_merge_patch()` function:

1. **Finds insertion point**: Locates the last occurrence of `car_sales_df.write`
2. **Preserves structure**: Keeps all code before and after
3. **Inserts snippet**: Adds the LLM suggestion before the write statement
4. **Maintains flow**: Ensures logical code execution order

```python
def smart_merge_patch(original_code: str, llm_snippet: str) -> str:
    """
    Intelligently merge an LLM code snippet into the original code.
    
    Strategy:
    1. Find the last occurrence of 'car_sales_df.write' in the original code
    2. Insert the LLM snippet before that line
    3. Keep all other code intact
    """
    lines = original_code.split('\n')
    
    # Find the last occurrence of 'car_sales_df.write'
    last_write_index = -1
    for i in range(len(lines) - 1, -1, -1):
        if 'car_sales_df.write' in lines[i]:
            last_write_index = i
            break
    
    # Insert the LLM snippet before the last write statement
    snippet_lines = llm_snippet.strip().split('\n')
    
    result_lines = (
        lines[:last_write_index] +  # All code before write
        snippet_lines +              # LLM suggestion
        [''] +                        # Blank line for readability
        lines[last_write_index:]      # All code after write
    )
    
    return '\n'.join(result_lines)
```

---

## Key Takeaways for RAG Function Improvement

1. **Context Preservation**: Don't replace, **merge**
2. **Intelligent Insertion**: Find the right place to insert code
3. **Code Structure Awareness**: Understand code flow and dependencies
4. **Minimal Changes**: Only modify what's necessary
5. **User Intent**: Preserve debugging code, comments, and structure

---

## Demo Script

**Opening**: "As an AI engineer, I want to show you a critical issue in how AI coding assistants handle code fixes."

**Problem Demo**: 
- Show original code → Error → LLM suggestion → Replace button → **Everything deleted** ❌

**Solution Demo**:
- Show original code → Error → LLM suggestion → Smart merge → **Everything preserved** ✅

**Conclusion**: 
- "The issue is that the RAG function treats LLM output as a complete replacement instead of a patch. We need to implement intelligent code merging that preserves context and structure."

