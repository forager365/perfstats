"""
Plotly Visualization Module for CPU Time Series Analysis
=========================================================
This module provides interactive visualizations using Plotly
for comparing CPU usage before and after server improvements.

Author: Statistical Analysis Tool
Usage: Import and call create_dashboard(before, after, time_labels)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional, Tuple
import plotly.express as px


def create_time_series_plot(
    before: np.ndarray,
    after: np.ndarray,
    time_labels: List[str],
    title: str = "CPU Usage Over 24 Hours: Before vs After"
) -> go.Figure:
    """
    Create an interactive time series comparison plot.
    
    Parameters:
    -----------
    before : np.ndarray
        CPU usage before improvement
    after : np.ndarray
        CPU usage after improvement
    time_labels : List[str]
        Time labels for x-axis
    title : str
        Plot title
        
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    fig = go.Figure()
    
    # Before trace
    fig.add_trace(go.Scatter(
        x=time_labels,
        y=before,
        mode='lines',
        name='Before',
        line=dict(color='#EF553B', width=2),
        hovertemplate='Time: %{x}<br>CPU: %{y:.2f}%<extra>Before</extra>'
    ))
    
    # After trace
    fig.add_trace(go.Scatter(
        x=time_labels,
        y=after,
        mode='lines',
        name='After',
        line=dict(color='#636EFA', width=2),
        hovertemplate='Time: %{x}<br>CPU: %{y:.2f}%<extra>After</extra>'
    ))
    
    # Fill between (improvement area)
    fig.add_trace(go.Scatter(
        x=time_labels + time_labels[::-1],
        y=list(before) + list(after[::-1]),
        fill='toself',
        fillcolor='rgba(0, 176, 80, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip',
        showlegend=True,
        name='Improvement'
    ))
    
    # Mean lines
    fig.add_hline(y=np.mean(before), line_dash="dash", line_color="#EF553B", 
                  annotation_text=f"Before Mean: {np.mean(before):.1f}%",
                  annotation_position="right")
    fig.add_hline(y=np.mean(after), line_dash="dash", line_color="#636EFA",
                  annotation_text=f"After Mean: {np.mean(after):.1f}%",
                  annotation_position="right")
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title="Time",
        yaxis_title="CPU Usage (%)",
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        template='plotly_white',
        height=500
    )
    
    # Update x-axis to show fewer labels
    fig.update_xaxes(
        tickmode='array',
        tickvals=time_labels[::8],
        tickangle=45
    )
    
    return fig


def create_box_plot(
    before: np.ndarray,
    after: np.ndarray,
    title: str = "Distribution Comparison"
) -> go.Figure:
    """
    Create an interactive box plot comparison.
    
    Parameters:
    -----------
    before : np.ndarray
        CPU usage before improvement
    after : np.ndarray
        CPU usage after improvement
    title : str
        Plot title
        
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    fig = go.Figure()
    
    fig.add_trace(go.Box(
        y=before,
        name='Before',
        boxpoints='outliers',
        marker_color='#EF553B',
        fillcolor='rgba(239, 85, 59, 0.5)',
        line=dict(color='#EF553B'),
        hovertemplate='CPU: %{y:.2f}%<extra>Before</extra>'
    ))
    
    fig.add_trace(go.Box(
        y=after,
        name='After',
        boxpoints='outliers',
        marker_color='#636EFA',
        fillcolor='rgba(99, 110, 250, 0.5)',
        line=dict(color='#636EFA'),
        hovertemplate='CPU: %{y:.2f}%<extra>After</extra>'
    ))
    
    # Add mean markers
    fig.add_trace(go.Scatter(
        x=['Before', 'After'],
        y=[np.mean(before), np.mean(after)],
        mode='markers',
        marker=dict(symbol='diamond', size=12, color='black'),
        name='Mean',
        hovertemplate='Mean: %{y:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        yaxis_title="CPU Usage (%)",
        template='plotly_white',
        showlegend=True,
        height=500
    )
    
    return fig


def create_histogram(
    before: np.ndarray,
    after: np.ndarray,
    title: str = "Distribution of CPU Usage"
) -> go.Figure:
    """
    Create an interactive overlapping histogram.
    
    Parameters:
    -----------
    before : np.ndarray
        CPU usage before improvement
    after : np.ndarray
        CPU usage after improvement
    title : str
        Plot title
        
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=before,
        name=f'Before (μ={np.mean(before):.1f}%)',
        opacity=0.6,
        marker_color='#EF553B',
        nbinsx=25,
        hovertemplate='CPU: %{x:.1f}%<br>Count: %{y}<extra>Before</extra>'
    ))
    
    fig.add_trace(go.Histogram(
        x=after,
        name=f'After (μ={np.mean(after):.1f}%)',
        opacity=0.6,
        marker_color='#636EFA',
        nbinsx=25,
        hovertemplate='CPU: %{x:.1f}%<br>Count: %{y}<extra>After</extra>'
    ))
    
    # Add vertical lines for means
    fig.add_vline(x=np.mean(before), line_dash="dash", line_color="#EF553B", line_width=2)
    fig.add_vline(x=np.mean(after), line_dash="dash", line_color="#636EFA", line_width=2)
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title="CPU Usage (%)",
        yaxis_title="Frequency",
        barmode='overlay',
        template='plotly_white',
        height=500
    )
    
    return fig


def create_difference_plot(
    before: np.ndarray,
    after: np.ndarray,
    time_labels: List[str],
    title: str = "Point-by-Point Improvement"
) -> go.Figure:
    """
    Create a bar chart showing the difference at each time point.
    
    Parameters:
    -----------
    before : np.ndarray
        CPU usage before improvement
    after : np.ndarray
        CPU usage after improvement
    time_labels : List[str]
        Time labels for x-axis
    title : str
        Plot title
        
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    difference = before - after
    colors = ['#00B050' if d > 0 else '#FF0000' for d in difference]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=time_labels,
        y=difference,
        marker_color=colors,
        name='Difference',
        hovertemplate='Time: %{x}<br>Improvement: %{y:.2f}%<extra></extra>'
    ))
    
    # Mean difference line
    fig.add_hline(y=np.mean(difference), line_dash="dash", line_color="#636EFA", line_width=2,
                  annotation_text=f"Mean Diff: {np.mean(difference):.2f}%",
                  annotation_position="right")
    
    # Zero line
    fig.add_hline(y=0, line_color="black", line_width=1)
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title="Time",
        yaxis_title="Difference (Before - After) %",
        template='plotly_white',
        height=500
    )
    
    fig.update_xaxes(
        tickmode='array',
        tickvals=time_labels[::8],
        tickangle=45
    )
    
    return fig


def create_violin_plot(
    before: np.ndarray,
    after: np.ndarray,
    title: str = "Violin Plot Comparison"
) -> go.Figure:
    """
    Create a violin plot for distribution comparison.
    
    Parameters:
    -----------
    before : np.ndarray
        CPU usage before improvement
    after : np.ndarray
        CPU usage after improvement
    title : str
        Plot title
        
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    fig = go.Figure()
    
    fig.add_trace(go.Violin(
        y=before,
        name='Before',
        box_visible=True,
        meanline_visible=True,
        fillcolor='rgba(239, 85, 59, 0.5)',
        line_color='#EF553B',
        points='outliers'
    ))
    
    fig.add_trace(go.Violin(
        y=after,
        name='After',
        box_visible=True,
        meanline_visible=True,
        fillcolor='rgba(99, 110, 250, 0.5)',
        line_color='#636EFA',
        points='outliers'
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        yaxis_title="CPU Usage (%)",
        template='plotly_white',
        height=500
    )
    
    return fig


def create_period_heatmap(
    before: np.ndarray,
    after: np.ndarray,
    title: str = "Improvement by Time Period"
) -> go.Figure:
    """
    Create a heatmap showing improvement by time period.
    
    Parameters:
    -----------
    before : np.ndarray
        CPU usage before improvement (96 points expected)
    after : np.ndarray
        CPU usage after improvement
    title : str
        Plot title
        
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    # Reshape to 24 hours x 4 quarters
    n_hours = 24
    n_quarters = 4
    
    before_reshaped = before.reshape(n_hours, n_quarters)
    after_reshaped = after.reshape(n_hours, n_quarters)
    improvement = before_reshaped - after_reshaped
    
    hours = [f'{h:02d}:00' for h in range(24)]
    quarters = ['00', '15', '30', '45']
    
    fig = go.Figure(data=go.Heatmap(
        z=improvement,
        x=quarters,
        y=hours,
        colorscale='RdYlGn',
        colorbar=dict(title='Improvement (%)'),
        hovertemplate='Hour: %{y}<br>Minute: %{x}<br>Improvement: %{z:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title="Minute of Hour",
        yaxis_title="Hour of Day",
        template='plotly_white',
        height=600
    )
    
    return fig


def create_qq_plot(
    before: np.ndarray,
    after: np.ndarray,
    title: str = "Q-Q Plot (Normality Check)"
) -> go.Figure:
    """
    Create Q-Q plots to check normality of the data.
    
    Parameters:
    -----------
    before : np.ndarray
        CPU usage before improvement
    after : np.ndarray
        CPU usage after improvement
    title : str
        Plot title
        
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    from scipy import stats
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Before', 'After'))
    
    # Before Q-Q
    sorted_before = np.sort(before)
    theoretical_quantiles_before = stats.norm.ppf(
        np.linspace(0.01, 0.99, len(before))
    )
    
    fig.add_trace(go.Scatter(
        x=theoretical_quantiles_before,
        y=sorted_before,
        mode='markers',
        marker=dict(color='#EF553B', size=6),
        name='Before',
        hovertemplate='Theoretical: %{x:.2f}<br>Actual: %{y:.2f}%<extra></extra>'
    ), row=1, col=1)
    
    # Reference line for before
    slope_b, intercept_b = np.polyfit(theoretical_quantiles_before, sorted_before, 1)
    fig.add_trace(go.Scatter(
        x=theoretical_quantiles_before,
        y=slope_b * theoretical_quantiles_before + intercept_b,
        mode='lines',
        line=dict(color='gray', dash='dash'),
        name='Reference',
        showlegend=False
    ), row=1, col=1)
    
    # After Q-Q
    sorted_after = np.sort(after)
    theoretical_quantiles_after = stats.norm.ppf(
        np.linspace(0.01, 0.99, len(after))
    )
    
    fig.add_trace(go.Scatter(
        x=theoretical_quantiles_after,
        y=sorted_after,
        mode='markers',
        marker=dict(color='#636EFA', size=6),
        name='After',
        hovertemplate='Theoretical: %{x:.2f}<br>Actual: %{y:.2f}%<extra></extra>'
    ), row=1, col=2)
    
    # Reference line for after
    slope_a, intercept_a = np.polyfit(theoretical_quantiles_after, sorted_after, 1)
    fig.add_trace(go.Scatter(
        x=theoretical_quantiles_after,
        y=slope_a * theoretical_quantiles_after + intercept_a,
        mode='lines',
        line=dict(color='gray', dash='dash'),
        name='Reference',
        showlegend=False
    ), row=1, col=2)
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        template='plotly_white',
        height=450
    )
    
    fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=1)
    fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
    fig.update_yaxes(title_text="Sample Quantiles", row=1, col=1)
    fig.update_yaxes(title_text="Sample Quantiles", row=1, col=2)
    
    return fig


def create_statistics_table(
    before: np.ndarray,
    after: np.ndarray,
    title: str = "Statistical Summary"
) -> go.Figure:
    """
    Create a styled table with statistical summary.
    
    Parameters:
    -----------
    before : np.ndarray
        CPU usage before improvement
    after : np.ndarray
        CPU usage after improvement
    title : str
        Table title
        
    Returns:
    --------
    go.Figure
        Plotly figure object with table
    """
    from scipy import stats
    
    # Calculate statistics
    metrics = ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'IQR', 'Skewness', 'Kurtosis']
    before_vals = [
        np.mean(before),
        np.median(before),
        np.std(before, ddof=1),
        np.min(before),
        np.max(before),
        np.percentile(before, 75) - np.percentile(before, 25),
        stats.skew(before),
        stats.kurtosis(before)
    ]
    after_vals = [
        np.mean(after),
        np.median(after),
        np.std(after, ddof=1),
        np.min(after),
        np.max(after),
        np.percentile(after, 75) - np.percentile(after, 25),
        stats.skew(after),
        stats.kurtosis(after)
    ]
    diff_vals = [b - a for b, a in zip(before_vals, after_vals)]
    
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>Metric</b>', '<b>Before</b>', '<b>After</b>', '<b>Difference</b>'],
            fill_color='#636EFA',
            font=dict(color='white', size=12),
            align='center',
            height=30
        ),
        cells=dict(
            values=[
                metrics,
                [f'{v:.2f}' for v in before_vals],
                [f'{v:.2f}' for v in after_vals],
                [f'{v:+.2f}' for v in diff_vals]
            ],
            fill_color=[
                ['white'] * len(metrics),
                ['#FFE6E6'] * len(metrics),
                ['#E6E6FF'] * len(metrics),
                ['#E6FFE6' if d > 0 else '#FFE6E6' for d in diff_vals[:4]] + ['white'] * 4
            ],
            align='center',
            height=25
        )
    )])
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        height=350
    )
    
    return fig


def create_dashboard(
    before: np.ndarray,
    after: np.ndarray,
    time_labels: List[str],
    title: str = "CPU Usage Analysis Dashboard"
) -> go.Figure:
    """
    Create a comprehensive dashboard with multiple visualizations.
    
    Parameters:
    -----------
    before : np.ndarray
        CPU usage before improvement
    after : np.ndarray
        CPU usage after improvement
    time_labels : List[str]
        Time labels for x-axis
    title : str
        Dashboard title
        
    Returns:
    --------
    go.Figure
        Plotly figure object with dashboard
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Time Series Comparison',
            'Distribution (Box Plot)',
            'Distribution (Histogram)',
            'Point-by-Point Improvement'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )
    
    # 1. Time Series (top-left)
    fig.add_trace(go.Scatter(
        x=time_labels, y=before,
        mode='lines', name='Before',
        line=dict(color='#EF553B', width=1.5),
        hovertemplate='%{x}: %{y:.1f}%<extra>Before</extra>'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=time_labels, y=after,
        mode='lines', name='After',
        line=dict(color='#636EFA', width=1.5),
        hovertemplate='%{x}: %{y:.1f}%<extra>After</extra>'
    ), row=1, col=1)
    
    # 2. Box Plot (top-right)
    fig.add_trace(go.Box(
        y=before, name='Before',
        marker_color='#EF553B',
        fillcolor='rgba(239, 85, 59, 0.5)',
        showlegend=False
    ), row=1, col=2)
    
    fig.add_trace(go.Box(
        y=after, name='After',
        marker_color='#636EFA',
        fillcolor='rgba(99, 110, 250, 0.5)',
        showlegend=False
    ), row=1, col=2)
    
    # 3. Histogram (bottom-left)
    fig.add_trace(go.Histogram(
        x=before, name='Before',
        marker_color='#EF553B', opacity=0.6,
        nbinsx=20, showlegend=False
    ), row=2, col=1)
    
    fig.add_trace(go.Histogram(
        x=after, name='After',
        marker_color='#636EFA', opacity=0.6,
        nbinsx=20, showlegend=False
    ), row=2, col=1)
    
    # 4. Difference Bar (bottom-right)
    difference = before - after
    colors = ['#00B050' if d > 0 else '#FF0000' for d in difference]
    
    fig.add_trace(go.Bar(
        x=time_labels, y=difference,
        marker_color=colors, name='Improvement',
        showlegend=False,
        hovertemplate='%{x}: %{y:.1f}%<extra></extra>'
    ), row=2, col=2)
    
    # Add mean difference line
    fig.add_hline(y=np.mean(difference), line_dash="dash", line_color="#636EFA",
                  row=2, col=2)
    
    # Update layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        height=800,
        template='plotly_white',
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        barmode='overlay'
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Time", row=1, col=1, tickangle=45, tickmode='array', tickvals=time_labels[::12])
    fig.update_yaxes(title_text="CPU (%)", row=1, col=1)
    fig.update_yaxes(title_text="CPU (%)", row=1, col=2)
    fig.update_xaxes(title_text="CPU (%)", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=2, tickangle=45, tickmode='array', tickvals=time_labels[::12])
    fig.update_yaxes(title_text="Difference (%)", row=2, col=2)
    
    return fig


def save_all_plots(
    before: np.ndarray,
    after: np.ndarray,
    time_labels: List[str],
    output_dir: str = ".",
    format: str = "html"
) -> List[str]:
    """
    Generate and save all plots to files.
    
    Parameters:
    -----------
    before : np.ndarray
        CPU usage before improvement
    after : np.ndarray
        CPU usage after improvement
    time_labels : List[str]
        Time labels for x-axis
    output_dir : str
        Directory to save plots
    format : str
        Output format ('html' or 'png')
        
    Returns:
    --------
    List[str]
        List of saved file paths
    """
    import os
    
    plots = {
        'time_series': create_time_series_plot(before, after, time_labels),
        'box_plot': create_box_plot(before, after),
        'histogram': create_histogram(before, after),
        'difference': create_difference_plot(before, after, time_labels),
        'violin': create_violin_plot(before, after),
        'heatmap': create_period_heatmap(before, after),
        'qq_plot': create_qq_plot(before, after),
        'stats_table': create_statistics_table(before, after),
        'dashboard': create_dashboard(before, after, time_labels)
    }
    
    saved_files = []
    for name, fig in plots.items():
        filepath = os.path.join(output_dir, f'{name}.{format}')
        if format == 'html':
            fig.write_html(filepath)
        else:
            fig.write_image(filepath, scale=2)
        saved_files.append(filepath)
        print(f"Saved: {filepath}")
    
    return saved_files


if __name__ == "__main__":
    # Example usage
    from datetime import datetime, timedelta
    
    np.random.seed(42)
    n = 96
    
    # Generate sample data
    time_intervals = [datetime(2024, 1, 1, 0, 0) + timedelta(minutes=15*i) for i in range(n)]
    time_labels = [t.strftime('%H:%M') for t in time_intervals]
    
    before = 50 + 20 * np.sin(np.linspace(0, 2*np.pi, n)) + np.random.normal(0, 8, n)
    after = 35 + 15 * np.sin(np.linspace(0, 2*np.pi, n)) + np.random.normal(0, 5, n)
    
    # Create and show dashboard
    dashboard = create_dashboard(before, after, time_labels)
    dashboard.show()
