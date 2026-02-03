#!/usr/bin/env python3
"""
CPU Usage Improvement Analysis - Main Script
=============================================
This script orchestrates the complete analysis workflow:
1. Generate or load CPU usage data
2. Perform statistical analysis
3. Generate interactive Plotly visualizations
4. Export results to files

Author: Statistical Analysis Tool
Usage: python main.py [--data-file <path>] [--output-dir <path>] [--seed <int>]
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

# Import project modules
from data_generator import (
    generate_cpu_data, 
    DataGenerationConfig,
    load_from_csv,
    save_to_csv
)
from statistical_analysis import (
    analyze_improvement, 
    print_analysis_report,
    AnalysisResults
)
from plotly_visualizations import (
    create_time_series_plot,
    create_box_plot,
    create_histogram,
    create_difference_plot,
    create_violin_plot,
    create_period_heatmap,
    create_qq_plot,
    create_statistics_table,
    create_dashboard,
    save_all_plots
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze CPU usage improvement between two time series.'
    )
    parser.add_argument(
        '--data-file', '-d',
        type=str,
        default=None,
        help='Path to CSV file with CPU data. If not provided, synthetic data is generated.'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./output',
        help='Directory for output files (default: ./output)'
    )
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=42,
        help='Random seed for reproducible synthetic data (default: 42)'
    )
    parser.add_argument(
        '--format', '-f',
        type=str,
        choices=['html', 'png', 'both'],
        default='html',
        help='Output format for plots (default: html)'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating visualizations'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.05,
        help='Significance level for statistical tests (default: 0.05)'
    )
    
    return parser.parse_args()


def create_output_directory(output_dir: str) -> None:
    """Create output directory if it doesn't exist."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")


def generate_summary_html(
    results: AnalysisResults,
    before: np.ndarray,
    after: np.ndarray,
    output_dir: str
) -> str:
    """
    Generate an HTML summary report.
    
    Parameters:
    -----------
    results : AnalysisResults
        Analysis results object
    before : np.ndarray
        Before data
    after : np.ndarray
        After data
    output_dir : str
        Output directory
        
    Returns:
    --------
    str
        Path to the generated HTML file
    """
    improvement_pct = results.improvement_pct
    ci = results.confidence_interval
    effect = results.effect_size
    
    # Determine verdict color
    if "YES" in results.verdict:
        verdict_color = "#00B050"
        verdict_icon = "✓"
    elif "MARGINAL" in results.verdict:
        verdict_color = "#FFA500"
        verdict_icon = "~"
    else:
        verdict_color = "#FF0000"
        verdict_icon = "✗"
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CPU Usage Improvement Analysis Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2em;
        }}
        .card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .card h2 {{
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .verdict {{
            font-size: 1.5em;
            font-weight: bold;
            color: {verdict_color};
            padding: 20px;
            text-align: center;
            background-color: rgba(0,0,0,0.05);
            border-radius: 10px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .metric-box {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        .metric-label {{
            color: #666;
            font-size: 0.9em;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #667eea;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .significant {{
            color: #00B050;
            font-weight: bold;
        }}
        .not-significant {{
            color: #FF0000;
        }}
        .plot-link {{
            display: inline-block;
            margin: 5px;
            padding: 10px 20px;
            background: #667eea;
            color: white;
            text-decoration: none;
            border-radius: 5px;
        }}
        .plot-link:hover {{
            background: #5a67d8;
        }}
        iframe {{
            width: 100%;
            height: 500px;
            border: none;
            border-radius: 8px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 CPU Usage Improvement Analysis</h1>
        <p>Statistical analysis of server performance before and after optimization</p>
    </div>
    
    <div class="card">
        <div class="verdict">
            {verdict_icon} {results.verdict}
        </div>
        <p style="text-align: center; color: #666; margin-top: 15px;">
            {results.verdict_details}
        </p>
    </div>
    
    <div class="card">
        <h2>📈 Key Metrics</h2>
        <div class="metrics-grid">
            <div class="metric-box">
                <div class="metric-value">{improvement_pct:.1f}%</div>
                <div class="metric-label">Mean Reduction</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{results.before_stats.mean:.1f}%</div>
                <div class="metric-label">Before (Mean)</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{results.after_stats.mean:.1f}%</div>
                <div class="metric-label">After (Mean)</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{effect.cohens_d:.2f}</div>
                <div class="metric-label">Cohen's d ({effect.interpretation})</div>
            </div>
        </div>
    </div>
    
    <div class="card">
        <h2>📋 Descriptive Statistics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Before</th>
                <th>After</th>
                <th>Difference</th>
            </tr>
            <tr>
                <td>Mean</td>
                <td>{results.before_stats.mean:.2f}%</td>
                <td>{results.after_stats.mean:.2f}%</td>
                <td>{results.before_stats.mean - results.after_stats.mean:.2f}%</td>
            </tr>
            <tr>
                <td>Median</td>
                <td>{results.before_stats.median:.2f}%</td>
                <td>{results.after_stats.median:.2f}%</td>
                <td>{results.before_stats.median - results.after_stats.median:.2f}%</td>
            </tr>
            <tr>
                <td>Std Dev</td>
                <td>{results.before_stats.std:.2f}%</td>
                <td>{results.after_stats.std:.2f}%</td>
                <td>{results.before_stats.std - results.after_stats.std:.2f}%</td>
            </tr>
            <tr>
                <td>Min</td>
                <td>{results.before_stats.min:.2f}%</td>
                <td>{results.after_stats.min:.2f}%</td>
                <td>-</td>
            </tr>
            <tr>
                <td>Max</td>
                <td>{results.before_stats.max:.2f}%</td>
                <td>{results.after_stats.max:.2f}%</td>
                <td>-</td>
            </tr>
        </table>
    </div>
    
    <div class="card">
        <h2>🔬 Statistical Tests</h2>
        <table>
            <tr>
                <th>Test</th>
                <th>Statistic</th>
                <th>p-value</th>
                <th>Result</th>
            </tr>
"""
    
    for test in results.tests:
        sig_class = "significant" if test.is_significant else "not-significant"
        sig_text = "Significant" if test.is_significant else "Not Significant"
        html_content += f"""            <tr>
                <td>{test.test_name}</td>
                <td>{test.statistic:.4f}</td>
                <td>{test.p_value:.6f}</td>
                <td class="{sig_class}">{sig_text}</td>
            </tr>
"""
    
    html_content += f"""        </table>
    </div>
    
    <div class="card">
        <h2>📊 Confidence Interval</h2>
        <p>
            <strong>Mean Difference:</strong> {ci.mean_diff:.2f}% (SE: {ci.std_error:.2f})<br>
            <strong>{ci.confidence_level*100:.0f}% Confidence Interval:</strong> 
            [{ci.lower:.2f}%, {ci.upper:.2f}%]
        </p>
        <p style="color: #666;">
            We are {ci.confidence_level*100:.0f}% confident that the true mean reduction 
            in CPU usage is between {ci.lower:.2f}% and {ci.upper:.2f}%.
        </p>
    </div>
"""
    
    # Add period analysis if available
    if results.period_analysis is not None:
        html_content += """    <div class="card">
        <h2>⏰ Time Period Analysis</h2>
        <table>
            <tr>
                <th>Period</th>
                <th>Before Mean</th>
                <th>After Mean</th>
                <th>Reduction</th>
                <th>p-value</th>
                <th>Significant</th>
            </tr>
"""
        for _, row in results.period_analysis.iterrows():
            sig_class = "significant" if row['Significant'] else "not-significant"
            sig_text = "Yes" if row['Significant'] else "No"
            html_content += f"""            <tr>
                <td>{row['Period']}</td>
                <td>{row['Before Mean']:.2f}%</td>
                <td>{row['After Mean']:.2f}%</td>
                <td>{row['Reduction']:.2f}% ({row['Reduction %']:.1f}%)</td>
                <td>{row['p-value']:.4f}</td>
                <td class="{sig_class}">{sig_text}</td>
            </tr>
"""
        html_content += """        </table>
    </div>
"""
    
    html_content += """    <div class="card">
        <h2>📊 Interactive Visualizations</h2>
        <p>Click to open individual plots:</p>
        <a href="dashboard.html" class="plot-link">📊 Full Dashboard</a>
        <a href="time_series.html" class="plot-link">📈 Time Series</a>
        <a href="box_plot.html" class="plot-link">📦 Box Plot</a>
        <a href="histogram.html" class="plot-link">📊 Histogram</a>
        <a href="difference.html" class="plot-link">📉 Difference Plot</a>
        <a href="violin.html" class="plot-link">🎻 Violin Plot</a>
        <a href="heatmap.html" class="plot-link">🗺️ Heatmap</a>
    </div>
    
    <div class="card">
        <h2>📊 Dashboard Preview</h2>
        <iframe src="dashboard.html"></iframe>
    </div>
    
    <footer style="text-align: center; color: #666; margin-top: 20px;">
        Generated by CPU Usage Analysis Tool
    </footer>
</body>
</html>
"""
    
    filepath = os.path.join(output_dir, 'report.html')
    with open(filepath, 'w') as f:
        f.write(html_content)
    
    return filepath


def main():
    """Main execution function."""
    args = parse_arguments()
    
    print("=" * 70)
    print("CPU USAGE IMPROVEMENT ANALYSIS")
    print("=" * 70)
    
    # Create output directory
    create_output_directory(args.output_dir)
    
    # Load or generate data
    if args.data_file:
        print(f"\nLoading data from: {args.data_file}")
        before, after, time_labels, df = load_from_csv(args.data_file)
    else:
        print(f"\nGenerating synthetic data (seed={args.seed})...")
        config = DataGenerationConfig(random_seed=args.seed)
        before, after, time_labels, df = generate_cpu_data(config)
        
        # Save synthetic data
        data_path = os.path.join(args.output_dir, 'cpu_data.csv')
        save_to_csv(before, after, time_labels, data_path)
    
    print(f"Data points: {len(before)}")
    print(f"Before: mean={np.mean(before):.2f}%, std={np.std(before):.2f}%")
    print(f"After:  mean={np.mean(after):.2f}%, std={np.std(after):.2f}%")
    
    # Perform statistical analysis
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)
    
    results = analyze_improvement(before, after, alpha=args.alpha)
    report = print_analysis_report(results)
    print(report)
    
    # Save text report
    report_path = os.path.join(args.output_dir, 'analysis_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nText report saved to: {report_path}")
    
    # Generate visualizations
    if not args.no_plots:
        print("\n" + "=" * 70)
        print("GENERATING VISUALIZATIONS")
        print("=" * 70)
        
        # Generate individual plots
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
        
        # Save plots
        for name, fig in plots.items():
            if args.format in ['html', 'both']:
                html_path = os.path.join(args.output_dir, f'{name}.html')
                fig.write_html(html_path)
                print(f"Saved: {html_path}")
            
            if args.format in ['png', 'both']:
                png_path = os.path.join(args.output_dir, f'{name}.png')
                fig.write_image(png_path, scale=2)
                print(f"Saved: {png_path}")
        
        # Generate HTML summary report
        html_report_path = generate_summary_html(results, before, after, args.output_dir)
        print(f"\nHTML report saved to: {html_report_path}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nAll outputs saved to: {args.output_dir}")
    print("\nKey files:")
    print(f"  - report.html      : Interactive HTML report")
    print(f"  - dashboard.html   : Full analysis dashboard")
    print(f"  - analysis_report.txt : Detailed text report")
    print(f"  - cpu_data.csv     : Raw data")
    
    return results


if __name__ == "__main__":
    main()
