# Slide Outline - Databricks Agent Code Merge Feedback

## Slide 1: Title
**Databricks Agent Code Merge Issue**
*Feedback from Hackathon Experience*
[Your Name] | [Date]

---

## Slide 2: Problem Statement
**The Issue:**
- Schema mismatch error (spaces vs underscores)
- Agent correctly identified fix ✅
- "Replace" button deleted ALL original code ❌
- Lost: setup, imports, debugging, display calls

**Impact:** Developer frustration, lost work, reduced trust in AI

---

## Slide 3: What Got Lost
**Original Code Had:**
- `spark.sql("USE dq_demo")` - Database context
- Variable definitions
- Import statements
- Print statements (debugging)
- Schema inspection
- Display calls

**After "Replace":**
- Only the column renaming snippet
- Everything else deleted

---

## Slide 4: Root Cause - Technical Analysis
**Current Behavior (Problematic):**
- RAG function treats LLM output as **complete replacement**
- No diff analysis or code structure understanding
- Direct string overwrite

**What's Missing:**
1. Patch/Diff mindset
2. Code structure awareness (AST parsing)
3. Intelligent insertion point detection
4. Context preservation

---

## Slide 5: Demo - Original Code
**[Screenshot of original code]**
- Show all components
- Highlight what will be lost

---

## Slide 6: Demo - Error & Suggestion
**[Screenshot of error + LLM suggestion]**
- Error clearly shows schema mismatch
- LLM suggests correct fix
- But suggestion is only a snippet

---

## Slide 7: Demo - Diff View
**[Screenshot of inline diff]**
- Green highlights show additions
- Visual preview of changes
- Like Cursor AI / GitHub

---

## Slide 8: Demo - Bad Behavior
**[Screenshot after "Replace"]**
- Only snippet remains
- All original code deleted
- This is current Databricks behavior

---

## Slide 9: Demo - Good Behavior
**[Screenshot after "Smart merge"]**
- All original code preserved
- Fix inserted at correct location
- Structure maintained

---

## Slide 10: Technical Implementation
**Smart Merge Algorithm:**
1. Find insertion point (pattern matching)
2. Preserve original code structure
3. Insert fix at correct location
4. Maintain formatting and indentation

**Diff Visualization:**
- `SequenceMatcher` for comparison
- Green/red highlighting
- Inline preview

---

## Slide 11: Key Takeaways
**What Needs to Change:**

1. **RAG Function Enhancement**
   - Treat LLM output as patches/diffs
   - Add code structure awareness
   - Implement intelligent merging

2. **User Experience**
   - Show diff preview before applying
   - Visual indicators (green/red)
   - Build user confidence

3. **Context Preservation**
   - Never delete original code
   - Preserve debugging/comments
   - Understand dependencies

---

## Slide 12: Benefits
**If Implemented:**
- ✅ Better user experience
- ✅ Reduced errors
- ✅ Faster development
- ✅ Increased trust in AI
- ✅ More adoption of AI features

---

## Slide 13: Next Steps
**Recommendations:**
1. Enhance RAG function with diff analysis
2. Add code structure awareness (AST parsing)
3. Implement intelligent merge algorithm
4. Add diff visualization UI
5. User testing and feedback

---

## Slide 14: Thank You
**Questions?**
[Your Contact Info]
[Demo Repository Link]

---

## Visual Elements Needed:
- Screenshots of each demo step
- Code comparison (before/after)
- Diagram of RAG function flow
- Architecture diagram of solution

