# Quick Demo Reference Card

## 🎯 One-Sentence Summary
**Problem**: Databricks Agent's "Replace" button deletes all original code when applying fixes.  
**Solution**: Smart merge patch preserves all code and inserts fixes intelligently.

---

## 📋 Demo Steps (2 minutes)

### 1. Show the Problem (30 seconds)
- Point to **"Current cell code"** → Original code with error
- Click **"Ask LLM for fix"** → Get suggestion
- Click **"Replace entire cell"** → Show how everything disappears ❌
- **Say**: "This is what happens in Databricks - all context is lost"

### 2. Show the Solution (30 seconds)
- Reset (or point to original code again)
- Click **"Ask LLM for fix"** → Same suggestion
- Click **"Smart merge patch"** → Show intelligent merge ✅
- **Say**: "Our solution preserves everything and inserts the fix correctly"

### 3. Explain the Root Cause (1 minute)
- **Problem**: RAG function treats LLM output as **complete replacement**
- **Solution**: Treat LLM output as **patch/diff** with intelligent merge
- **Key**: Find insertion point (before `car_sales_df.write`) and preserve context

---

## 💡 Key Talking Points

### The Issue
- CSV columns have **spaces** (`"Engine size"`)
- Delta table has **underscores** (`"Engine_size"`)
- Schema mismatch error occurs
- Agent suggests correct fix (column renaming)
- **BUT**: "Replace" button deletes ALL original code

### What Gets Lost
- Database setup: `spark.sql("USE dq_demo")`
- Variable definitions: `car_csv_path = "..."`
- Import statements: `from pyspark.sql.functions import col`
- Debugging: `print()` statements
- Schema inspection: `printSchema()`
- Data preview: `display()`

### The Fix
- **Smart merge** finds insertion point (before write statement)
- Preserves ALL original code
- Inserts only the necessary fix
- Maintains code structure and flow

---

## 🔑 Technical Details (If Asked)

### Smart Merge Algorithm
1. Find last occurrence of `car_sales_df.write`
2. Insert LLM snippet before that line
3. Preserve all code before and after
4. Maintain logical execution order

### RAG Function Improvement Needed
- **Current**: LLM output → Complete replacement
- **Needed**: LLM output → Patch/diff → Intelligent merge
- **Key**: Context awareness and code structure understanding

---

## 📊 Expected Output Comparison

### ❌ Bad (Current Databricks Behavior)
```python
# Only the snippet - everything else deleted
car_sales_df = (
    car_sales_df
    .withColumnRenamed("Engine size", "Engine_size")
    ...
)
car_sales_df.write.mode("overwrite").saveAsTable("silver_car_sales")
```

### ✅ Good (Smart Merge)
```python
# All original code preserved + fix inserted
spark.sql("USE dq_demo")
car_csv_path = "..."
car_sales_df = spark.read...
from pyspark.sql.functions import col

# Fix inserted here
car_sales_df = (
    car_sales_df
    .withColumnRenamed("Engine size", "Engine_size")
    ...
)

car_sales_df.write.mode("overwrite").saveAsTable("silver_car_sales")
print("✅ Created table...")
car_sales_df.printSchema()
display(car_sales_df)
# ... all original code preserved
```

---

## 🎤 Closing Statement

"As an AI engineer, this demo shows a critical gap in how AI coding assistants handle code fixes. The RAG function needs to understand that LLM suggestions are **patches**, not **replacements**. We need intelligent code merging that preserves context, structure, and user intent."

