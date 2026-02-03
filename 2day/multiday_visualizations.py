"""
Multi-Day Plotly Visualizations
================================
Interactive visualizations for multi-day CPU usage analysis.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Tuple


def create_multiday_time_series(
    before_data: np.ndarray,
    after_data: np.ndarray,
    title: str = "CPU Usage: All Days Comparison"
) -> go.Figure:
    """
    Create time series showing all days overlaid.
    
    Parameters:
    -----------
    before_data : np.ndarray
        Shape (n_days, points_per_day)
    after_data : np.ndarray
        Shape (n_days, points_per_day)
    """
    n_days_before, n_points = before_data.shape
    n_days_after = after_data.shape[0]
    
    # Generate time labels
    time_labels = [f'{(i*15)//60:02d}:{(i*15)%60:02d}' for i in range(n_points)]
    
    fig = go.Figure()
    
    # Plot each "before" day
    colors_before = ['#EF553B', '#FF7F7F', '#FFB3B3']  # Shades of red
    for day in range(n_days_before):
        fig.add_trace(go.Scatter(
            x=time_labels,
            y=before_data[day],
            mode='lines',
            name=f'Before Day {day+1}',
            line=dict(color=colors_before[day % len(colors_before)], width=1.5),
            legendgroup='before',
            hovertemplate=f'Day {day+1}<br>Time: %{{x}}<br>CPU: %{{y:.1f}}%<extra>Before</extra>'
        ))
    
    # Plot each "after" day
    colors_after = ['#636EFA', '#7F7FFF', '#B3B3FF']  # Shades of blue
    for day in range(n_days_after):
        fig.add_trace(go.Scatter(
            x=time_labels,
            y=after_data[day],
            mode='lines',
            name=f'After Day {day+1}',
            line=dict(color=colors_after[day % len(colors_after)], width=1.5),
            legendgroup='after',
            hovertemplate=f'Day {day+1}<br>Time: %{{x}}<br>CPU: %{{y:.1f}}%<extra>After</extra>'
        ))
    
    # Add mean lines
    fig.add_hline(y=np.mean(before_data), line_dash="dash", line_color="#EF553B",
                  annotation_text=f"Before Mean: {np.mean(before_data):.1f}%")
    fig.add_hline(y=np.mean(after_data), line_dash="dash", line_color="#636EFA",
                  annotation_text=f"After Mean: {np.mean(after_data):.1f}%")
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title="Time of Day",
        yaxis_title="CPU Usage (%)",
        template='plotly_white',
        height=500,
        hovermode='x unified'
    )
    
    fig.update_xaxes(tickmode='array', tickvals=time_labels[::8], tickangle=45)
    
    return fig


def create_averaged_comparison(
    before_data: np.ndarray,
    after_data: np.ndarray,
    title: str = "Averaged Daily Profile with Confidence Bands"
) -> go.Figure:
    """
    Show averaged profiles with standard error bands.
    """
    n_points = before_data.shape[1]
    time_labels = [f'{(i*15)//60:02d}:{(i*15)%60:02d}' for i in range(n_points)]
    
    # Calculate means and standard errors
    before_mean = np.mean(before_data, axis=0)
    before_se = np.std(before_data, axis=0, ddof=1) / np.sqrt(before_data.shape[0])
    
    after_mean = np.mean(after_data, axis=0)
    after_se = np.std(after_data, axis=0, ddof=1) / np.sqrt(after_data.shape[0])
    
    fig = go.Figure()
    
    # Before confidence band
    fig.add_trace(go.Scatter(
        x=time_labels + time_labels[::-1],
        y=list(before_mean + 1.96*before_se) + list((before_mean - 1.96*before_se)[::-1]),
        fill='toself',
        fillcolor='rgba(239, 85, 59, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Before 95% CI',
        showlegend=True,
        hoverinfo='skip'
    ))
    
    # After confidence band
    fig.add_trace(go.Scatter(
        x=time_labels + time_labels[::-1],
        y=list(after_mean + 1.96*after_se) + list((after_mean - 1.96*after_se)[::-1]),
        fill='toself',
        fillcolor='rgba(99, 110, 250, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='After 95% CI',
        showlegend=True,
        hoverinfo='skip'
    ))
    
    # Mean lines
    fig.add_trace(go.Scatter(
        x=time_labels, y=before_mean,
        mode='lines', name='Before (Mean)',
        line=dict(color='#EF553B', width=2.5)
    ))
    
    fig.add_trace(go.Scatter(
        x=time_labels, y=after_mean,
        mode='lines', name='After (Mean)',
        line=dict(color='#636EFA', width=2.5)
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title="Time of Day",
        yaxis_title="CPU Usage (%)",
        template='plotly_white',
        height=500
    )
    
    fig.update_xaxes(tickmode='array', tickvals=time_labels[::8], tickangle=45)
    
    return fig


def create_daily_means_comparison(
    before_data: np.ndarray,
    after_data: np.ndarray,
    title: str = "Daily Mean Comparison"
) -> go.Figure:
    """
    Bar chart comparing daily means.
    """
    before_daily = np.mean(before_data, axis=1)
    after_daily = np.mean(after_data, axis=1)
    
    n_before = len(before_daily)
    n_after = len(after_daily)
    
    fig = go.Figure()
    
    # Before bars
    fig.add_trace(go.Bar(
        x=[f'Before Day {i+1}' for i in range(n_before)],
        y=before_daily,
        name='Before',
        marker_color='#EF553B',
        text=[f'{v:.1f}%' for v in before_daily],
        textposition='outside'
    ))
    
    # After bars
    fig.add_trace(go.Bar(
        x=[f'After Day {i+1}' for i in range(n_after)],
        y=after_daily,
        name='After',
        marker_color='#636EFA',
        text=[f'{v:.1f}%' for v in after_daily],
        textposition='outside'
    ))
    
    # Add mean lines
    fig.add_hline(y=np.mean(before_daily), line_dash="dash", line_color="#EF553B",
                  annotation_text=f"Before Avg: {np.mean(before_daily):.1f}%")
    fig.add_hline(y=np.mean(after_daily), line_dash="dash", line_color="#636EFA",
                  annotation_text=f"After Avg: {np.mean(after_daily):.1f}%")
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title="Day",
        yaxis_title="Mean CPU Usage (%)",
        template='plotly_white',
        height=450,
        barmode='group'
    )
    
    return fig


def create_variance_decomposition(
    before_data: np.ndarray,
    after_data: np.ndarray,
    title: str = "Variance Decomposition"
) -> go.Figure:
    """
    Show within-day vs between-day variance.
    """
    # Calculate variance components
    # Between-day variance
    var_between_before = np.var(np.mean(before_data, axis=1), ddof=1)
    var_between_after = np.var(np.mean(after_data, axis=1), ddof=1)
    
    # Within-day variance (average)
    var_within_before = np.mean([np.var(before_data[d], ddof=1) for d in range(before_data.shape[0])])
    var_within_after = np.mean([np.var(after_data[d], ddof=1) for d in range(after_data.shape[0])])
    
    fig = go.Figure()
    
    categories = ['Before', 'After']
    
    fig.add_trace(go.Bar(
        name='Between-Day Variance',
        x=categories,
        y=[var_between_before, var_between_after],
        marker_color='#FF6B6B',
        text=[f'{v:.1f}' for v in [var_between_before, var_between_after]],
        textposition='inside'
    ))
    
    fig.add_trace(go.Bar(
        name='Within-Day Variance',
        x=categories,
        y=[var_within_before, var_within_after],
        marker_color='#4ECDC4',
        text=[f'{v:.1f}' for v in [var_within_before, var_within_after]],
        textposition='inside'
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title="Condition",
        yaxis_title="Variance",
        template='plotly_white',
        height=400,
        barmode='stack'
    )
    
    return fig


def create_approach_comparison_chart(
    results: List,
    title: str = "Comparison of Statistical Approaches"
) -> go.Figure:
    """
    Create a chart comparing results from different approaches.
    
    Parameters:
    -----------
    results : List[MultiDayResults]
        Results from run_all_approaches()
    """
    approaches = [r.approach for r in results]
    improvements = [r.improvement_pct for r in results]
    p_values = [r.p_value for r in results]
    effect_sizes = [r.effect_size for r in results]
    significant = [r.is_significant for r in results]
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Improvement (%)', 'P-value', "Effect Size (Cohen's d)")
    )
    
    colors = ['#00B050' if s else '#FF6B6B' for s in significant]
    
    # Improvement
    fig.add_trace(go.Bar(
        x=approaches, y=improvements,
        marker_color=colors,
        text=[f'{v:.1f}%' for v in improvements],
        textposition='outside',
        showlegend=False
    ), row=1, col=1)
    
    # P-values (log scale)
    fig.add_trace(go.Bar(
        x=approaches, y=p_values,
        marker_color=colors,
        text=[f'{v:.4f}' for v in p_values],
        textposition='outside',
        showlegend=False
    ), row=1, col=2)
    
    # Add significance threshold line
    fig.add_hline(y=0.05, line_dash="dash", line_color="red", row=1, col=2,
                  annotation_text="α = 0.05")
    
    # Effect sizes
    fig.add_trace(go.Bar(
        x=approaches, y=effect_sizes,
        marker_color=colors,
        text=[f'{v:.2f}' for v in effect_sizes],
        textposition='outside',
        showlegend=False
    ), row=1, col=3)
    
    # Add effect size thresholds
    fig.add_hline(y=0.2, line_dash="dot", line_color="gray", row=1, col=3)
    fig.add_hline(y=0.5, line_dash="dot", line_color="gray", row=1, col=3)
    fig.add_hline(y=0.8, line_dash="dot", line_color="gray", row=1, col=3)
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        template='plotly_white',
        height=450,
        showlegend=False
    )
    
    fig.update_xaxes(tickangle=45)
    
    return fig


def create_multiday_dashboard(
    before_data: np.ndarray,
    after_data: np.ndarray,
    results: List = None,
    title: str = "Multi-Day CPU Analysis Dashboard"
) -> go.Figure:
    """
    Create comprehensive dashboard for multi-day analysis.
    """
    n_points = before_data.shape[1]
    time_labels = [f'{(i*15)//60:02d}:{(i*15)%60:02d}' for i in range(n_points)]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'All Days Overlay',
            'Averaged Profile with CI',
            'Daily Means',
            'Distribution Comparison'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # 1. All days overlay (top-left)
    for day in range(before_data.shape[0]):
        fig.add_trace(go.Scatter(
            x=time_labels, y=before_data[day],
            mode='lines', name=f'Before D{day+1}',
            line=dict(color='#EF553B', width=1),
            legendgroup='before',
            showlegend=(day == 0)
        ), row=1, col=1)
    
    for day in range(after_data.shape[0]):
        fig.add_trace(go.Scatter(
            x=time_labels, y=after_data[day],
            mode='lines', name=f'After D{day+1}',
            line=dict(color='#636EFA', width=1),
            legendgroup='after',
            showlegend=(day == 0)
        ), row=1, col=1)
    
    # 2. Averaged with CI (top-right)
    before_mean = np.mean(before_data, axis=0)
    after_mean = np.mean(after_data, axis=0)
    
    fig.add_trace(go.Scatter(
        x=time_labels, y=before_mean,
        mode='lines', name='Before Mean',
        line=dict(color='#EF553B', width=2),
        showlegend=False
    ), row=1, col=2)
    
    fig.add_trace(go.Scatter(
        x=time_labels, y=after_mean,
        mode='lines', name='After Mean',
        line=dict(color='#636EFA', width=2),
        showlegend=False
    ), row=1, col=2)
    
    # 3. Daily means (bottom-left)
    before_daily = np.mean(before_data, axis=1)
    after_daily = np.mean(after_data, axis=1)
    
    x_before = [f'B-D{i+1}' for i in range(len(before_daily))]
    x_after = [f'A-D{i+1}' for i in range(len(after_daily))]
    
    fig.add_trace(go.Bar(
        x=x_before, y=before_daily,
        marker_color='#EF553B',
        name='Before',
        showlegend=False
    ), row=2, col=1)
    
    fig.add_trace(go.Bar(
        x=x_after, y=after_daily,
        marker_color='#636EFA',
        name='After',
        showlegend=False
    ), row=2, col=1)
    
    # 4. Box plots (bottom-right)
    fig.add_trace(go.Box(
        y=before_data.flatten(),
        name='Before',
        marker_color='#EF553B',
        showlegend=False
    ), row=2, col=2)
    
    fig.add_trace(go.Box(
        y=after_data.flatten(),
        name='After',
        marker_color='#636EFA',
        showlegend=False
    ), row=2, col=2)
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        height=700,
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    
    # Update axes
    for col in [1, 2]:
        fig.update_xaxes(tickangle=45, tickmode='array', 
                        tickvals=time_labels[::12], row=1, col=col)
    
    return fig


if __name__ == "__main__":
    from multiday_analysis import generate_multiday_data
    
    # Generate sample data
    before_data, after_data, df = generate_multiday_data(
        n_days_before=2,
        n_days_after=2,
        random_seed=42
    )
    
    # Create and show dashboard
    fig = create_multiday_dashboard(before_data, after_data)
    fig.write_html('multiday_dashboard.html')
    print("Dashboard saved to: multiday_dashboard.html")
