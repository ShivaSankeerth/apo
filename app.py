"""
APO Streamlit App - Interactive Prompt Optimization

This app provides a user-friendly interface for optimizing prompts using the APO framework.
"""

import streamlit as st
import asyncio
import json
import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict, Any
from datetime import datetime

# Import APO components
from apo import MetaPromptOptimizer, ReflectionOptimizer, AnthropicProvider, OpenAIProvider
from apo.core import Evaluator, EvaluationResult, Prompt


# ============================================================================
# Custom Evaluator using user feedback
# ============================================================================

class UserFeedbackEvaluator(Evaluator):
    """Evaluator that uses user feedback to score prompts."""

    def __init__(self, provider, feedback_data: Dict[str, Any]):
        super().__init__()
        self.provider = provider
        self.feedback_data = feedback_data

    async def evaluate(self, prompt: Prompt, test_cases: List[Dict[str, Any]]) -> EvaluationResult:
        """Evaluate prompt based on test cases and user feedback."""
        total_score = 0.0
        results = []

        for test_case in test_cases:
            # Render prompt with test case
            rendered = self._render_prompt(prompt.content, test_case)

            # Get LLM response
            response = await self.provider.complete(rendered, temperature=0.7)
            output = response.content.strip()

            # Score based on user feedback if available
            test_id = test_case.get('id', '')
            if test_id in self.feedback_data:
                score = self.feedback_data[test_id]
            else:
                # Default neutral score
                score = 0.5

            total_score += score
            results.append({
                'input': test_case,
                'output': output,
                'score': score
            })

        avg_score = total_score / len(test_cases) if test_cases else 0

        return EvaluationResult(
            prompt=prompt,
            metrics={'score': avg_score},
            metadata={'results': results}
        )

    def _render_prompt(self, prompt_template: str, test_case: Dict[str, Any]) -> str:
        """Render prompt with test case variables."""
        rendered = prompt_template
        for key, value in test_case.items():
            if key != 'id':
                placeholder = f"{{{key}}}"
                rendered = rendered.replace(placeholder, str(value))
        return rendered


# ============================================================================
# Session State Initialization
# ============================================================================

def init_session_state():
    """Initialize session state variables."""
    if 'test_cases' not in st.session_state:
        st.session_state.test_cases = []
    if 'current_results' not in st.session_state:
        st.session_state.current_results = []
    if 'feedback' not in st.session_state:
        st.session_state.feedback = {}
    if 'optimization_history' not in st.session_state:
        st.session_state.optimization_history = []
    if 'current_prompt' not in st.session_state:
        st.session_state.current_prompt = ""
    if 'optimized_prompt' not in st.session_state:
        st.session_state.optimized_prompt = None
    if 'provider' not in st.session_state:
        st.session_state.provider = None


# ============================================================================
# Helper Functions
# ============================================================================

async def run_prompt_async(provider, prompt: str, test_cases: List[Dict]) -> List[Dict]:
    """Run prompt against test cases and return results."""
    results = []
    for i, test_case in enumerate(test_cases):
        # Render prompt
        rendered = prompt
        for key, value in test_case.items():
            if key != 'id':
                rendered = rendered.replace(f"{{{key}}}", str(value))

        # Get response
        response = await provider.complete(rendered, temperature=0.7, max_tokens=500)

        results.append({
            'id': test_case.get('id', f"test_{i}"),
            'input': test_case,
            'output': response.content.strip(),
            'prompt_used': rendered
        })

    return results


def run_prompt(provider, prompt: str, test_cases: List[Dict]) -> List[Dict]:
    """Synchronous wrapper for running prompts."""
    return asyncio.run(run_prompt_async(provider, prompt, test_cases))


async def optimize_prompt_async(
    provider,
    initial_prompt: str,
    test_cases: List[Dict],
    feedback: Dict[str, float],
    strategy: str = "meta"
) -> Dict[str, Any]:
    """Optimize prompt using APO framework."""
    # Create evaluator with feedback
    evaluator = UserFeedbackEvaluator(provider, feedback)

    # Choose optimizer
    if strategy == "meta":
        optimizer = MetaPromptOptimizer(
            provider=provider,
            evaluator=evaluator,
            max_iterations=5,
            patience=2,
            verbose=False
        )
    else:  # reflection
        optimizer = ReflectionOptimizer(
            provider=provider,
            evaluator=evaluator,
            use_pareto_frontier=True,
            max_iterations=5,
            patience=2,
            verbose=False
        )

    # Run optimization
    result = await optimizer.optimize(
        initial_prompt=initial_prompt,
        test_cases=test_cases
    )

    return {
        'best_prompt': result.best_prompt.content,
        'best_score': result.best_score,
        'history': [
            {
                'score': eval_result.score,
                'prompt': eval_result.prompt.content[:100] + "..."
            }
            for eval_result in result.history
        ],
        'total_evaluations': result.total_evaluations,
        'duration': result.duration
    }


def optimize_prompt(
    provider,
    initial_prompt: str,
    test_cases: List[Dict],
    feedback: Dict[str, float],
    strategy: str = "meta"
) -> Dict[str, Any]:
    """Synchronous wrapper for optimization."""
    return asyncio.run(optimize_prompt_async(provider, initial_prompt, test_cases, feedback, strategy))


# ============================================================================
# UI Components
# ============================================================================

def render_sidebar():
    """Render sidebar with configuration."""
    with st.sidebar:
        st.title("⚙️ Configuration")

        # API Provider Selection
        provider_type = st.selectbox(
            "LLM Provider",
            ["Anthropic (Claude)", "OpenAI (GPT)"],
            help="Select your LLM provider"
        )

        # API Key
        api_key = st.text_input(
            "API Key",
            type="password",
            help="Enter your API key"
        )

        # Model Selection
        if provider_type == "Anthropic (Claude)":
            model = st.selectbox(
                "Model",
                ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-sonnet-20240229"]
            )
        else:
            model = st.selectbox(
                "Model",
                ["gpt-4-turbo-preview", "gpt-4", "gpt-3.5-turbo"]
            )

        # Optimization Strategy
        st.markdown("---")
        st.subheader("Optimization Strategy")
        strategy = st.radio(
            "Choose optimizer",
            ["meta", "reflection"],
            format_func=lambda x: "Meta-Prompt (Arize-inspired)" if x == "meta" else "Reflection (GEPA-inspired)",
            help="Meta-Prompt: Fast, feedback-driven. Reflection: Explores multiple strategies."
        )

        # Initialize provider
        if api_key:
            try:
                if provider_type == "Anthropic (Claude)":
                    st.session_state.provider = AnthropicProvider(api_key=api_key, model=model)
                else:
                    st.session_state.provider = OpenAIProvider(api_key=api_key, model=model)
                st.success("✅ Provider initialized!")
            except Exception as e:
                st.error(f"Error initializing provider: {str(e)}")
                st.session_state.provider = None
        else:
            st.warning("⚠️ Please enter your API key")

        return strategy


def render_prompt_input():
    """Render prompt input section."""
    st.header("1️⃣ Define Your Prompt")

    col1, col2 = st.columns([3, 1])

    with col1:
        prompt = st.text_area(
            "Initial Prompt",
            value=st.session_state.current_prompt,
            height=150,
            placeholder="Enter your prompt here. Use {variable} for placeholders.\n\nExample: Classify the sentiment of the following text: {text}",
            help="Use curly braces {} for variables that will be replaced with test case values"
        )

    with col2:
        st.markdown("**Tips:**")
        st.markdown("- Use `{variable}` for inputs")
        st.markdown("- Be clear and specific")
        st.markdown("- Include examples if helpful")

    st.session_state.current_prompt = prompt

    # Show detected variables
    import re
    variables = re.findall(r'\{(\w+)\}', prompt)
    if variables:
        st.info(f"📝 Detected variables: {', '.join(set(variables))}")


def render_test_cases():
    """Render test case management."""
    st.header("2️⃣ Add Test Cases")

    # Add new test case
    with st.expander("➕ Add New Test Case", expanded=len(st.session_state.test_cases) == 0):
        # Detect variables from prompt
        import re
        variables = list(set(re.findall(r'\{(\w+)\}', st.session_state.current_prompt)))

        if not variables:
            st.warning("⚠️ No variables detected in prompt. Add {variable} placeholders first.")
            return

        # Create input fields for each variable
        test_case = {}
        cols = st.columns(len(variables))
        for i, var in enumerate(variables):
            with cols[i]:
                test_case[var] = st.text_input(f"{var}", key=f"new_{var}")

        if st.button("Add Test Case", type="primary"):
            if all(test_case.values()):
                test_case['id'] = f"test_{len(st.session_state.test_cases)}"
                st.session_state.test_cases.append(test_case)
                st.success("✅ Test case added!")
                st.rerun()
            else:
                st.error("Please fill all fields")

    # Display existing test cases
    if st.session_state.test_cases:
        st.subheader(f"Test Cases ({len(st.session_state.test_cases)})")

        for i, test_case in enumerate(st.session_state.test_cases):
            col1, col2 = st.columns([4, 1])
            with col1:
                # Display test case
                display_text = " | ".join([f"{k}: {v}" for k, v in test_case.items() if k != 'id'])
                st.text(f"{i+1}. {display_text}")
            with col2:
                if st.button("🗑️", key=f"delete_{i}"):
                    st.session_state.test_cases.pop(i)
                    st.rerun()


def render_test_run():
    """Render test run section."""
    st.header("3️⃣ Test Your Prompt")

    if not st.session_state.provider:
        st.warning("⚠️ Please configure your API provider in the sidebar")
        return

    if not st.session_state.test_cases:
        st.warning("⚠️ Please add test cases first")
        return

    if st.button("▶️ Run Tests", type="primary", use_container_width=True):
        with st.spinner("Running tests..."):
            try:
                results = run_prompt(
                    st.session_state.provider,
                    st.session_state.current_prompt,
                    st.session_state.test_cases
                )
                st.session_state.current_results = results
                st.success(f"✅ Completed {len(results)} tests!")
            except Exception as e:
                st.error(f"Error running tests: {str(e)}")

    # Display results
    if st.session_state.current_results:
        st.subheader("Results")

        for i, result in enumerate(st.session_state.current_results):
            with st.expander(f"Test {i+1}", expanded=True):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Input:**")
                    for key, value in result['input'].items():
                        if key != 'id':
                            st.text(f"{key}: {value}")

                with col2:
                    st.markdown("**Output:**")
                    st.markdown(result['output'])

                # Feedback
                st.markdown("**Your Feedback:**")
                feedback_col1, feedback_col2, feedback_col3 = st.columns(3)

                with feedback_col1:
                    if st.button("👍 Good", key=f"good_{i}"):
                        st.session_state.feedback[result['id']] = 1.0
                        st.rerun()

                with feedback_col2:
                    if st.button("😐 Okay", key=f"okay_{i}"):
                        st.session_state.feedback[result['id']] = 0.5
                        st.rerun()

                with feedback_col3:
                    if st.button("👎 Bad", key=f"bad_{i}"):
                        st.session_state.feedback[result['id']] = 0.0
                        st.rerun()

                # Show current feedback
                if result['id'] in st.session_state.feedback:
                    score = st.session_state.feedback[result['id']]
                    emoji = "👍" if score >= 0.8 else "😐" if score >= 0.4 else "👎"
                    st.info(f"{emoji} Feedback: {score:.1%}")


def render_optimization():
    """Render optimization section."""
    st.header("4️⃣ Optimize Your Prompt")

    if not st.session_state.feedback:
        st.info("💡 Provide feedback on the test results above to enable optimization")
        return

    # Show feedback summary
    feedback_count = len(st.session_state.feedback)
    avg_score = sum(st.session_state.feedback.values()) / feedback_count if feedback_count > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Test Cases with Feedback", feedback_count)
    col2.metric("Average Score", f"{avg_score:.1%}")
    col3.metric("Total Tests", len(st.session_state.test_cases))

    strategy = st.session_state.get('strategy', 'meta')

    if st.button("🚀 Optimize Prompt", type="primary", use_container_width=True):
        with st.spinner("Optimizing prompt... This may take a minute..."):
            try:
                optimization_result = optimize_prompt(
                    st.session_state.provider,
                    st.session_state.current_prompt,
                    st.session_state.test_cases,
                    st.session_state.feedback,
                    strategy=strategy
                )

                st.session_state.optimized_prompt = optimization_result['best_prompt']
                st.session_state.optimization_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'original': st.session_state.current_prompt,
                    'optimized': optimization_result['best_prompt'],
                    'score_improvement': optimization_result['best_score'] - avg_score,
                    'evaluations': optimization_result['total_evaluations'],
                    'duration': optimization_result['duration'],
                    'history': optimization_result['history']
                })

                st.success("✅ Optimization complete!")
                st.balloons()
            except Exception as e:
                st.error(f"Error during optimization: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    # Show optimized prompt
    if st.session_state.optimized_prompt:
        st.subheader("📊 Optimization Results")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Original Prompt:**")
            st.code(st.session_state.current_prompt, language=None)

        with col2:
            st.markdown("**Optimized Prompt:**")
            st.code(st.session_state.optimized_prompt, language=None)

        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Use Optimized Prompt", use_container_width=True):
                st.session_state.current_prompt = st.session_state.optimized_prompt
                st.session_state.optimized_prompt = None
                st.session_state.current_results = []
                st.session_state.feedback = {}
                st.success("Prompt updated! Run tests again to see improvements.")
                st.rerun()

        with col2:
            if st.button("🔄 Try Again", use_container_width=True):
                st.session_state.optimized_prompt = None
                st.rerun()

        # Show optimization history
        if st.session_state.optimization_history:
            last_opt = st.session_state.optimization_history[-1]

            st.markdown("---")
            st.subheader("Optimization Progress")

            # Create progress chart
            if 'history' in last_opt and last_opt['history']:
                history_df = pd.DataFrame(last_opt['history'])
                history_df['iteration'] = range(len(history_df))

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=history_df['iteration'],
                    y=history_df['score'],
                    mode='lines+markers',
                    name='Score',
                    line=dict(color='#00cc96', width=3),
                    marker=dict(size=8)
                ))

                fig.update_layout(
                    title="Optimization Progress",
                    xaxis_title="Iteration",
                    yaxis_title="Score",
                    hovermode='x unified',
                    height=300
                )

                st.plotly_chart(fig, use_container_width=True)

            # Show metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Evaluations", last_opt['evaluations'])
            col2.metric("Duration", f"{last_opt['duration']:.2f}s")
            col3.metric("Score Improvement", f"{last_opt['score_improvement']:.2%}")


# ============================================================================
# Main App
# ============================================================================

def main():
    """Main application."""
    st.set_page_config(
        page_title="APO - Prompt Optimizer",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
        <style>
        .stButton button {
            border-radius: 8px;
        }
        .stExpander {
            border-radius: 8px;
            border: 1px solid #e0e0e0;
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize
    init_session_state()

    # Title
    st.title("🎯 APO - Automatic Prompt Optimizer")
    st.markdown("Optimize your prompts using AI-powered feedback and state-of-the-art algorithms")

    # Sidebar
    strategy = render_sidebar()
    st.session_state.strategy = strategy

    # Main content
    st.markdown("---")

    # Render sections
    render_prompt_input()
    st.markdown("---")
    render_test_cases()
    st.markdown("---")
    render_test_run()
    st.markdown("---")
    render_optimization()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with APO Framework | Inspired by Arize Phoenix & DSPy/GEPA</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
