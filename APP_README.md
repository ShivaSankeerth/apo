# APO Streamlit App

Interactive web application for optimizing prompts using the APO framework.

## Installation

1. Install APO framework:
```bash
pip install -e .
```

2. Install app dependencies:
```bash
pip install -r requirements-app.txt
```

## Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Features

### 1. Prompt Definition
- Write your initial prompt with variable placeholders
- Use `{variable}` syntax for dynamic inputs
- Real-time variable detection

### 2. Test Case Management
- Add multiple test cases with different inputs
- Edit and delete test cases
- Visual organization

### 3. Interactive Testing
- Run your prompt against all test cases
- See outputs in real-time
- Provide feedback (Good/Okay/Bad) on each result

### 4. AI-Powered Optimization
- Choose between optimization strategies:
  - **Meta-Prompt (Arize-inspired)**: Fast, feedback-driven optimization
  - **Reflection (GEPA-inspired)**: Explores multiple strategies with Pareto frontier
- See optimization progress in real-time
- Compare original vs optimized prompts
- View improvement metrics

### 5. Iteration
- Use optimized prompt for next iteration
- Track optimization history
- Continuous improvement cycle

## Usage Example

1. **Configure** (Sidebar):
   - Select LLM provider (Anthropic or OpenAI)
   - Enter your API key
   - Choose model and optimization strategy

2. **Define Prompt**:
   ```
   Classify the sentiment of the following text as positive, negative, or neutral: {text}
   ```

3. **Add Test Cases**:
   - Text: "I love this product!"
   - Text: "This is terrible."
   - Text: "It's okay, nothing special."

4. **Run Tests**:
   - Click "Run Tests"
   - Review outputs
   - Provide feedback on each result

5. **Optimize**:
   - Click "Optimize Prompt"
   - Wait for AI to improve your prompt
   - Review the optimized version
   - Use it or try again

## Optimization Strategies

### Meta-Prompt Optimizer
- Uses LLM reflection on feedback
- Identifies what works and what doesn't
- Generates improved prompts
- Best for: Quick iterations with clear feedback

### Reflection Optimizer
- Analyzes patterns in successes/failures
- Maintains Pareto frontier of strategies
- Explores multiple improvement directions
- Best for: Complex tasks with multiple objectives

## Tips for Best Results

1. **Clear Prompts**: Start with a clear, specific prompt
2. **Good Test Cases**: Use diverse, representative test cases
3. **Honest Feedback**: Provide accurate feedback on results
4. **Multiple Iterations**: Run optimization multiple times for best results
5. **Experiment**: Try different strategies to see what works best

## Architecture

The app uses:
- **Frontend**: Streamlit for interactive UI
- **Backend**: APO framework for optimization
- **Evaluation**: Custom `UserFeedbackEvaluator` that uses your feedback
- **Visualization**: Plotly for progress charts

## Troubleshooting

### API Key Issues
- Ensure your API key is valid
- Check you have sufficient credits
- Verify the correct provider is selected

### Slow Performance
- Reduce number of test cases (5-10 is optimal)
- Use faster models (claude-3-sonnet, gpt-3.5-turbo)
- Lower max_iterations in code if needed

### Poor Optimization Results
- Provide more test cases
- Give more varied feedback
- Try the other optimization strategy
- Make initial prompt more specific

## Development

To modify the app:

1. Edit `app.py` for UI changes
2. Modify optimizer parameters in the code:
   ```python
   optimizer = MetaPromptOptimizer(
       provider=provider,
       evaluator=evaluator,
       max_iterations=5,  # Adjust this
       patience=2,         # And this
       verbose=False
   )
   ```

3. Add new features by extending the session state and UI components

## Contributing

Feel free to enhance the app with:
- More visualization options
- Export/import functionality
- Batch optimization
- Custom evaluation metrics
- Multi-user support
