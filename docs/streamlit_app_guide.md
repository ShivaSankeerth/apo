# APO Streamlit App - User Guide

## Overview

The APO Streamlit app provides an intuitive, interactive interface for optimizing prompts using AI-powered feedback and cutting-edge optimization algorithms.

## Quick Start

```bash
# Install dependencies
pip install -r requirements-app.txt

# Run the app
streamlit run app.py
```

The app will launch at `http://localhost:8501`

## Workflow

### Step 1: Configure Your Provider

**Location**: Sidebar

1. **Select LLM Provider**:
   - Anthropic (Claude) - Recommended for best results
   - OpenAI (GPT)

2. **Enter API Key**:
   - Your API key (stored in session, not saved)
   - Get keys from: https://console.anthropic.com or https://platform.openai.com

3. **Choose Model**:
   - Claude: `claude-3-5-sonnet-20241022` (recommended)
   - GPT: `gpt-4-turbo-preview`

4. **Select Optimization Strategy**:
   - **Meta-Prompt** (Arize-inspired): Fast, uses feedback to reflect and improve
   - **Reflection** (GEPA-inspired): Explores multiple strategies with Pareto frontier

### Step 2: Define Your Prompt

**Location**: Main area - Section 1

1. Write your prompt in the text area
2. Use `{variable}` syntax for dynamic inputs
3. Example:
   ```
   Classify the sentiment of the following text as positive, negative, or neutral.

   Text: {text}
   Sentiment:
   ```

**Tips**:
- Be clear and specific
- Include instructions and context
- Use consistent variable names

### Step 3: Add Test Cases

**Location**: Main area - Section 2

1. Click "Add New Test Case" expander
2. The app automatically detects variables from your prompt
3. Fill in values for each variable
4. Click "Add Test Case"
5. Repeat for multiple test cases (5-10 recommended)

**Example Test Cases** (for sentiment classification):
```
Test 1: {text} = "I love this product! Best purchase ever!"
Test 2: {text} = "This is terrible. Waste of money."
Test 3: {text} = "It's okay, nothing special."
Test 4: {text} = "Amazing! Highly recommend!"
Test 5: {text} = "Very disappointed with the quality."
```

### Step 4: Test Your Prompt

**Location**: Main area - Section 3

1. Click "▶️ Run Tests"
2. Wait for results (may take 10-30 seconds)
3. Review each output
4. **Provide Feedback** on each result:
   - 👍 **Good**: Output is exactly what you want (Score: 100%)
   - 😐 **Okay**: Output is acceptable but could be better (Score: 50%)
   - 👎 **Bad**: Output is incorrect or unhelpful (Score: 0%)

**Why Feedback Matters**:
Your feedback teaches the optimizer what "good" looks like. The more accurate your feedback, the better the optimization results.

### Step 5: Optimize Your Prompt

**Location**: Main area - Section 4

1. After providing feedback on at least some test cases, click "🚀 Optimize Prompt"
2. The optimizer will:
   - Analyze your feedback patterns
   - Identify what's working and what isn't
   - Generate improved prompt variations
   - Evaluate them against your test cases
   - Select the best performing version
3. View optimization progress chart
4. Compare original vs optimized prompt
5. See performance metrics

**Results**:
- **Original Prompt**: Your starting prompt
- **Optimized Prompt**: AI-improved version
- **Score Improvement**: How much better it performed
- **Evaluations**: Number of prompts tested
- **Duration**: Time taken

### Step 6: Iterate

After optimization:

**Option A - Use Optimized Prompt**:
1. Click "✅ Use Optimized Prompt"
2. The optimized version becomes your new starting point
3. Run tests again to verify improvements
4. Provide feedback and optimize further

**Option B - Try Again**:
1. Click "🔄 Try Again"
2. Keeps current prompt
3. Runs optimization again (may explore different directions)

## Optimization Strategies Explained

### Meta-Prompt Optimizer (Recommended for Beginners)

**How it works**:
1. Analyzes your prompt and feedback
2. Uses an LLM to reflect on what's working/not working
3. Generates an improved prompt based on insights
4. Iterates 5 times or until no improvement

**Best for**:
- Quick iterations
- Clear, specific tasks
- When you have strong feedback signals
- General prompt improvement

**Example Use Case**:
Optimizing a customer support response template where you know exactly what good responses look like.

### Reflection Optimizer (Advanced)

**How it works**:
1. LLM reflects on successes and failures
2. Generates multiple improvement directions
3. Maintains a Pareto frontier of complementary strategies
4. Explores different approaches simultaneously
5. Selects best overall strategy

**Best for**:
- Complex tasks with multiple objectives
- When trade-offs exist (e.g., accuracy vs brevity)
- Exploring different prompt strategies
- Maximum optimization quality

**Example Use Case**:
Optimizing a summarization prompt where you want both accuracy AND conciseness.

## Tips for Best Results

### 1. Start with a Decent Prompt
- Don't start with "summarize {text}" - too vague
- Do start with "Provide a 2-sentence summary of the following text, focusing on key facts: {text}"

### 2. Use Diverse Test Cases
- Cover different scenarios
- Include edge cases
- Mix easy and hard examples
- Aim for 5-10 test cases

### 3. Provide Honest Feedback
- Don't mark bad outputs as good
- Use "Okay" for borderline cases
- Be consistent in your criteria

### 4. Iterate Multiple Times
- First optimization: 20-30% improvement typical
- Second optimization: Additional 10-15%
- Keep going until satisfied

### 5. Experiment with Strategies
- Try Meta-Prompt first (faster)
- Try Reflection for complex tasks
- Compare results

## Example Workflows

### Workflow 1: Sentiment Classification

```
1. Initial Prompt:
   "Classify: {text}"

2. Test Cases:
   - "I love this!" → Should be: positive
   - "This is bad" → Should be: negative
   - "It's okay" → Should be: neutral

3. After Testing:
   - Initial accuracy: 60% (too vague)
   - Provide feedback on errors

4. After Optimization:
   "Classify the sentiment of the following text as exactly one of: positive, negative, or neutral.

   Text: {text}

   Sentiment:"

   - New accuracy: 90%
```

### Workflow 2: Question Answering

```
1. Initial Prompt:
   "Answer: {question}"

2. Issues:
   - Too verbose
   - Sometimes incorrect
   - Includes unnecessary explanations

3. After Optimization:
   "Provide a concise, factual answer to the following question. Answer in one sentence only.

   Question: {question}
   Answer:"

   - More concise and accurate
```

### Workflow 3: Code Generation

```
1. Initial Prompt:
   "Write code for: {task}"

2. Issues:
   - Missing error handling
   - No comments
   - Not following best practices

3. After Optimization:
   "Write clean, production-ready Python code for the following task. Include:
   - Proper error handling
   - Type hints
   - Docstrings
   - Comments for complex logic

   Task: {task}

   Code:"

   - Much better quality code
```

## Troubleshooting

### "Error initializing provider"
- Check API key is correct
- Verify you have API credits
- Check internet connection

### "Error running tests"
- Ensure all test case fields are filled
- Check prompt has matching variables
- Try with fewer test cases
- Verify API key has sufficient permissions

### "Error during optimization"
- Provide feedback on at least 3 test cases
- Ensure feedback is varied (not all good or all bad)
- Try reducing test cases to 5-7
- Switch to a different optimization strategy

### Optimization not improving prompt
- Provide more specific feedback
- Add more diverse test cases
- Try the other optimization strategy
- Make your initial prompt more specific
- Run optimization multiple times

### Slow performance
- Use faster models (claude-3-sonnet instead of opus)
- Reduce number of test cases to 5-7
- Close other browser tabs
- Check your internet speed

## Advanced Usage

### Custom Evaluator
You can modify the `UserFeedbackEvaluator` class in `app.py` to add custom scoring logic:

```python
class UserFeedbackEvaluator(Evaluator):
    async def evaluate(self, prompt: Prompt, test_cases: List[Dict[str, Any]]):
        # Add custom metrics
        # For example: length penalty, keyword matching, etc.
        pass
```

### Adjusting Optimization Parameters

In `app.py`, modify the optimizer initialization:

```python
optimizer = MetaPromptOptimizer(
    provider=provider,
    evaluator=evaluator,
    max_iterations=10,    # More iterations = better results, slower
    patience=5,           # How long to wait for improvement
    verbose=True          # See detailed logs
)
```

### Export/Import Functionality

The app stores data in `st.session_state`. You can add export functionality:

```python
# Export
import json
export_data = {
    'prompt': st.session_state.current_prompt,
    'test_cases': st.session_state.test_cases,
    'feedback': st.session_state.feedback
}
st.download_button("Export", json.dumps(export_data))

# Import
uploaded = st.file_uploader("Import")
if uploaded:
    data = json.load(uploaded)
    st.session_state.current_prompt = data['prompt']
    # etc.
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit UI                         │
│  ┌────────────┐  ┌──────────────┐  ┌────────────────┐ │
│  │  Sidebar   │  │ Main Content │  │   Feedback     │ │
│  │  Config    │  │  Sections    │  │   Collection   │ │
│  └────────────┘  └──────────────┘  └────────────────┘ │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              APO Framework Backend                      │
│  ┌──────────────────┐      ┌─────────────────────────┐ │
│  │ UserFeedback     │      │  Optimization Strategy  │ │
│  │ Evaluator        │◄────►│  - MetaPromptOptimizer  │ │
│  │                  │      │  - ReflectionOptimizer  │ │
│  └──────────────────┘      └─────────────────────────┘ │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                 LLM Providers                           │
│  ┌──────────────────┐      ┌─────────────────────────┐ │
│  │  Anthropic       │      │      OpenAI             │ │
│  │  (Claude)        │      │      (GPT)              │ │
│  └──────────────────┘      └─────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the APP_README.md file
3. Check the main APO documentation in docs/
4. File an issue on GitHub

## Contributing

Ideas for enhancements:
- Add batch import of test cases (CSV/JSON)
- Save/load prompt sessions
- A/B testing between prompts
- Custom evaluation metrics
- Multi-language support
- Prompt templates library
- Collaboration features
