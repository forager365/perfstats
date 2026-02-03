# CPU Usage Improvement Analysis Tool

A comprehensive Python toolkit for statistically analyzing CPU usage time series data to determine if server improvements have resulted in measurable performance gains.

## Features

- **Statistical Analysis**: Paired t-tests, Wilcoxon signed-rank tests, Mann-Whitney U tests, effect size calculations
- **Interactive Visualizations**: Plotly-based dashboards and individual plots
- **Synthetic Data Generation**: Realistic CPU usage patterns for testing
- **Multiple Output Formats**: HTML reports, interactive plots, PNG images, CSV data
- **Time Period Analysis**: Breakdown by time of day (night, morning, afternoon, evening)

## Installation

```bash
# Clone or copy the project
cd cpu_analysis_project

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Using Synthetic Data

```bash
# Generate and analyze synthetic data
python main.py --seed 42 --output-dir ./results
```

### 2. Using Your Own Data

Prepare a CSV file with columns: `Time`, `CPU_Before (%)`, `CPU_After (%)`

```bash
python main.py --data-file your_data.csv --output-dir ./results
```

### 3. Python API Usage

```python
from data_generator import generate_cpu_data, DataGenerationConfig
from statistical_analysis import analyze_improvement, print_analysis_report
from plotly_visualizations import create_dashboard

# Generate or load data
config = DataGenerationConfig(random_seed=42)
before, after, time_labels, df = generate_cpu_data(config)

# Run statistical analysis
results = analyze_improvement(before, after, alpha=0.05)
print(print_analysis_report(results))

# Create interactive dashboard
fig = create_dashboard(before, after, time_labels)
fig.show()  # Opens in browser
fig.write_html('dashboard.html')  # Save to file
```

## Project Structure

```
cpu_analysis_project/
├── main.py                    # Main script - orchestrates full analysis
├── statistical_analysis.py    # Statistical tests and metrics
├── plotly_visualizations.py   # Interactive Plotly visualizations
├── data_generator.py          # Synthetic data generation
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Module Details

### 1. `statistical_analysis.py`

Core statistical functions:

| Function | Description |
|----------|-------------|
| `analyze_improvement()` | Complete analysis with all tests |
| `paired_t_test()` | Compare corresponding time points |
| `wilcoxon_test()` | Non-parametric paired comparison |
| `mann_whitney_test()` | Non-parametric independent comparison |
| `calculate_effect_size()` | Cohen's d and Glass's delta |
| `calculate_confidence_interval()` | 95% CI for mean difference |
| `analyze_by_time_period()` | Breakdown by time of day |

### 2. `plotly_visualizations.py`

Interactive visualization functions:

| Function | Description |
|----------|-------------|
| `create_dashboard()` | Complete 4-panel dashboard |
| `create_time_series_plot()` | Before/After overlay with improvement area |
| `create_box_plot()` | Distribution comparison |
| `create_histogram()` | Overlapping histograms |
| `create_difference_plot()` | Point-by-point improvement bars |
| `create_violin_plot()` | Violin plot comparison |
| `create_period_heatmap()` | Heatmap by hour/minute |
| `create_qq_plot()` | Q-Q plots for normality check |

### 3. `data_generator.py`

Synthetic data generation:

| Function | Description |
|----------|-------------|
| `generate_cpu_data()` | Standard improvement scenario |
| `generate_no_improvement_data()` | Null hypothesis testing |
| `generate_marginal_improvement_data()` | Small effect size |
| `generate_large_improvement_data()` | Large effect size |
| `generate_variable_improvement_data()` | Time-varying improvement |

## Command Line Options

```bash
python main.py [OPTIONS]

Options:
  -d, --data-file PATH    CSV file with CPU data (optional)
  -o, --output-dir PATH   Output directory (default: ./output)
  -s, --seed INT          Random seed for synthetic data (default: 42)
  -f, --format FORMAT     Output format: html, png, or both (default: html)
  --no-plots              Skip generating visualizations
  --alpha FLOAT           Significance level (default: 0.05)
```

## CSV Data Format

Your input CSV should have these columns:

```csv
Time,CPU_Before (%),CPU_After (%)
00:00,45.23,32.15
00:15,48.91,35.67
00:30,42.55,30.22
...
```

## Statistical Methodology

### Tests Performed

1. **Normality Check** (Shapiro-Wilk): Determines if parametric tests are appropriate
2. **Paired t-test**: Compares means at corresponding time points
3. **Wilcoxon Signed-Rank**: Non-parametric alternative to paired t-test
4. **Independent t-test**: Compares overall distributions
5. **Mann-Whitney U**: Non-parametric comparison (one-tailed)
6. **Levene's Test**: Checks equality of variances

### Effect Size Interpretation

| Cohen's d | Interpretation |
|-----------|---------------|
| < 0.2 | Negligible |
| 0.2 - 0.5 | Small |
| 0.5 - 0.8 | Medium |
| > 0.8 | Large |

### Decision Framework

A significant improvement requires:
- p-value < α (default 0.05) from paired t-test
- Cohen's d > 0.2 (at least small effect)
- Consistent results across non-parametric tests

## Example Output

### Verdict Types

✓ **YES - Significant Improvement**: Statistically significant with meaningful effect size
~ **MARGINAL**: Statistically significant but small practical effect
✗ **NO - No Significant Improvement**: No statistically significant difference

### Sample Results

```
VERDICT: ✓ YES - Significant Improvement Detected

Mean CPU usage decreased from 54.47% to 38.67%
This represents a 29.0% relative reduction
Cohen's d = 0.86 indicates a LARGE effect size
95% CI for reduction: [13.58%, 18.01%]
```

## Customization

### Custom Time Periods

```python
from statistical_analysis import analyze_by_time_period

# Define custom periods (indices for 96-point data)
custom_periods = {
    'Early Morning (00:00-08:00)': (0, 32),
    'Work Hours (08:00-17:00)': (32, 68),
    'Evening (17:00-24:00)': (68, 96)
}

period_df = analyze_by_time_period(before, after, periods=custom_periods)
```

### Custom Data Configuration

```python
from data_generator import DataGenerationConfig, generate_cpu_data

config = DataGenerationConfig(
    n_points=96,                    # 15-min intervals for 24 hours
    base_cpu_before=60.0,           # Base CPU before
    base_cpu_after=40.0,            # Base CPU after (improvement)
    noise_std_before=10.0,          # Noise level before
    noise_std_after=6.0,            # Reduced noise after
    spike_probability_before=0.2,   # Spike frequency before
    spike_probability_after=0.05,   # Reduced spikes after
    random_seed=42                  # Reproducibility
)

before, after, time_labels, df = generate_cpu_data(config)
```

## License

MIT License - feel free to use and modify for your needs.

## Contributing

Feel free to submit issues and enhancement requests!
