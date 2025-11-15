"""Visualization utilities for benchmark results."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import List
from benchmarks.runner import BenchmarkResult


def create_performance_comparison(results: List[BenchmarkResult], save_path: str = None) -> go.Figure:
    """Create bar chart comparing optimizer performance."""
    df = pd.DataFrame([
        {
            'Optimizer': r.optimizer_name,
            'Task': r.task_name,
            'Initial Score': r.initial_score,
            'Final Score': r.final_score,
            'Improvement': r.improvement,
        }
        for r in results
    ])

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=list(df['Task'].unique()),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    tasks = df['Task'].unique()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    for idx, task in enumerate(tasks):
        row = idx // 2 + 1
        col = idx % 2 + 1

        task_df = df[df['Task'] == task].sort_values('Final Score', ascending=False)

        fig.add_trace(
            go.Bar(
                name='Initial',
                x=task_df['Optimizer'],
                y=task_df['Initial Score'],
                marker_color='lightgray',
                showlegend=(idx == 0),
                text=[f"{v:.1%}" for v in task_df['Initial Score']],
                textposition='outside',
            ),
            row=row, col=col
        )

        fig.add_trace(
            go.Bar(
                name='Final',
                x=task_df['Optimizer'],
                y=task_df['Final Score'],
                marker_color='#2ca02c',
                showlegend=(idx == 0),
                text=[f"{v:.1%}" for v in task_df['Final Score']],
                textposition='outside',
            ),
            row=row, col=col
        )

    fig.update_xaxes(tickangle=45)
    fig.update_yaxes(title_text="Accuracy", range=[0, 1.1])

    fig.update_layout(
        title_text="Optimizer Performance by Task",
        height=800,
        showlegend=True,
        barmode='group',
        font=dict(size=10)
    )

    if save_path:
        fig.write_html(save_path)

    return fig


def create_convergence_plot(results: List[BenchmarkResult], task_name: str, save_path: str = None) -> go.Figure:
    """Create convergence plot for a specific task."""
    task_results = [r for r in results if r.task_name == task_name]

    fig = go.Figure()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    for idx, result in enumerate(task_results):
        if result.score_history:
            fig.add_trace(go.Scatter(
                x=list(range(len(result.score_history))),
                y=result.score_history,
                mode='lines+markers',
                name=result.optimizer_name,
                line=dict(width=2, color=colors[idx % len(colors)]),
                marker=dict(size=6),
                hovertemplate='%{y:.2%}<extra></extra>'
            ))

    fig.update_layout(
        title=f"Convergence Curves - {task_name}",
        xaxis_title="Iteration",
        yaxis_title="Score",
        hovermode='x unified',
        height=500,
        legend=dict(x=1.05, y=1)
    )

    fig.update_yaxes(range=[0, 1.1])

    if save_path:
        fig.write_html(save_path)

    return fig


def create_efficiency_plot(results: List[BenchmarkResult], save_path: str = None) -> go.Figure:
    """Create scatter plot of improvement vs time/evaluations."""
    df = pd.DataFrame([
        {
            'Optimizer': r.optimizer_name,
            'Task': r.task_name,
            'Improvement': r.improvement,
            'Duration': r.duration_seconds,
            'Evaluations': r.total_evaluations,
            'Final Score': r.final_score,
        }
        for r in results
    ])

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Improvement vs Time', 'Improvement vs Evaluations'),
        horizontal_spacing=0.15
    )

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    optimizers = df['Optimizer'].unique()

    for idx, optimizer in enumerate(optimizers):
        opt_df = df[df['Optimizer'] == optimizer]

        # Time efficiency
        fig.add_trace(
            go.Scatter(
                x=opt_df['Duration'],
                y=opt_df['Improvement'],
                mode='markers',
                name=optimizer,
                marker=dict(size=12, color=colors[idx % len(colors)]),
                text=opt_df['Task'],
                hovertemplate='<b>%{text}</b><br>Time: %{x:.1f}s<br>Improvement: %{y:.2%}<extra></extra>',
                showlegend=True,
            ),
            row=1, col=1
        )

        # Evaluation efficiency
        fig.add_trace(
            go.Scatter(
                x=opt_df['Evaluations'],
                y=opt_df['Improvement'],
                mode='markers',
                name=optimizer,
                marker=dict(size=12, color=colors[idx % len(colors)]),
                text=opt_df['Task'],
                hovertemplate='<b>%{text}</b><br>Evals: %{x}<br>Improvement: %{y:.2%}<extra></extra>',
                showlegend=False,
            ),
            row=1, col=2
        )

    fig.update_xaxes(title_text="Duration (seconds)", row=1, col=1)
    fig.update_xaxes(title_text="Total Evaluations", row=1, col=2)
    fig.update_yaxes(title_text="Improvement", row=1, col=1)
    fig.update_yaxes(title_text="Improvement", row=1, col=2)

    fig.update_layout(
        title_text="Optimizer Efficiency Analysis",
        height=500,
        hovermode='closest'
    )

    if save_path:
        fig.write_html(save_path)

    return fig


def create_summary_table(results: List[BenchmarkResult]) -> pd.DataFrame:
    """Create summary statistics table."""
    df = pd.DataFrame([
        {
            'Optimizer': r.optimizer_name,
            'Task': r.task_name,
            'Initial': f"{r.initial_score:.1%}",
            'Final': f"{r.final_score:.1%}",
            'Improvement': f"{r.improvement:+.1%}",
            'Time (s)': f"{r.duration_seconds:.1f}",
            'Evaluations': r.total_evaluations,
            'Iterations': r.iterations,
        }
        for r in results
    ])

    return df


def create_winner_chart(results: List[BenchmarkResult], save_path: str = None) -> go.Figure:
    """Create chart showing which optimizer won on each task."""
    # Find best optimizer for each task
    tasks = {}
    for result in results:
        if result.task_name not in tasks:
            tasks[result.task_name] = []
        tasks[result.task_name].append(result)

    winners = []
    for task_name, task_results in tasks.items():
        best = max(task_results, key=lambda x: x.final_score)
        winners.append({
            'Task': task_name,
            'Winner': best.optimizer_name,
            'Score': best.final_score,
            'Improvement': best.improvement
        })

    df = pd.DataFrame(winners)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df['Task'],
        y=df['Score'],
        text=[f"{opt}<br>{score:.1%}" for opt, score in zip(df['Winner'], df['Score'])],
        textposition='inside',
        marker=dict(
            color=df['Score'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Score")
        ),
        hovertemplate='<b>%{x}</b><br>Winner: %{text}<br>Score: %{y:.2%}<extra></extra>'
    ))

    fig.update_layout(
        title="Best Optimizer by Task",
        xaxis_title="Task",
        yaxis_title="Final Score",
        height=400,
        yaxis=dict(range=[0, 1.1])
    )

    if save_path:
        fig.write_html(save_path)

    return fig


def generate_all_visualizations(results: List[BenchmarkResult], output_dir: str = "benchmark_results"):
    """Generate all visualizations and save to directory."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Performance comparison
    create_performance_comparison(results, f"{output_dir}/performance_comparison.html")

    # Convergence plots for each task
    tasks = set(r.task_name for r in results)
    for task in tasks:
        safe_name = task.replace(" ", "_").lower()
        create_convergence_plot(results, task, f"{output_dir}/convergence_{safe_name}.html")

    # Efficiency analysis
    create_efficiency_plot(results, f"{output_dir}/efficiency_analysis.html")

    # Winner chart
    create_winner_chart(results, f"{output_dir}/winners.html")

    # Summary table
    summary_df = create_summary_table(results)
    summary_df.to_csv(f"{output_dir}/summary.csv", index=False)
    summary_df.to_html(f"{output_dir}/summary.html", index=False)

    print(f"All visualizations saved to {output_dir}/")
