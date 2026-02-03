"""
Statistical Analysis Module for CPU Time Series Comparison
===========================================================
This module provides statistical tests and metrics to determine
if there is a significant improvement between two time series.

Author: Statistical Analysis Tool
Usage: Import and call analyze_improvement(before, after)
"""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional


@dataclass
class DescriptiveStats:
    """Container for descriptive statistics"""
    mean: float
    median: float
    std: float
    min: float
    max: float
    range: float
    iqr: float
    cv: float  # Coefficient of variation
    q25: float
    q75: float


@dataclass
class TestResult:
    """Container for statistical test results"""
    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    interpretation: str


@dataclass
class EffectSize:
    """Container for effect size metrics"""
    cohens_d: float
    glass_delta: float
    interpretation: str


@dataclass
class ConfidenceInterval:
    """Container for confidence interval"""
    mean_diff: float
    std_error: float
    lower: float
    upper: float
    confidence_level: float


@dataclass
class AnalysisResults:
    """Complete analysis results container"""
    before_stats: DescriptiveStats
    after_stats: DescriptiveStats
    improvement_pct: float
    tests: List[TestResult]
    effect_size: EffectSize
    confidence_interval: ConfidenceInterval
    period_analysis: Optional[pd.DataFrame]
    verdict: str
    verdict_details: str


def calculate_descriptive_stats(data: np.ndarray) -> DescriptiveStats:
    """
    Calculate comprehensive descriptive statistics for a dataset.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data array
        
    Returns:
    --------
    DescriptiveStats
        Dataclass containing all descriptive statistics
    """
    q25 = np.percentile(data, 25)
    q75 = np.percentile(data, 75)
    
    return DescriptiveStats(
        mean=np.mean(data),
        median=np.median(data),
        std=np.std(data, ddof=1),
        min=np.min(data),
        max=np.max(data),
        range=np.max(data) - np.min(data),
        iqr=q75 - q25,
        cv=(np.std(data, ddof=1) / np.mean(data)) * 100,
        q25=q25,
        q75=q75
    )


def test_normality(data: np.ndarray, alpha: float = 0.05) -> TestResult:
    """
    Perform Shapiro-Wilk test for normality.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data array
    alpha : float
        Significance level (default 0.05)
        
    Returns:
    --------
    TestResult
        Test results with interpretation
    """
    stat, p_value = stats.shapiro(data)
    is_normal = p_value > alpha
    
    return TestResult(
        test_name="Shapiro-Wilk Normality Test",
        statistic=stat,
        p_value=p_value,
        is_significant=not is_normal,
        interpretation="Normal distribution" if is_normal else "Non-normal distribution"
    )


def paired_t_test(before: np.ndarray, after: np.ndarray, alpha: float = 0.05) -> TestResult:
    """
    Perform paired t-test comparing corresponding time points.
    
    Parameters:
    -----------
    before : np.ndarray
        Before improvement data
    after : np.ndarray
        After improvement data
    alpha : float
        Significance level
        
    Returns:
    --------
    TestResult
        Test results with interpretation
    """
    stat, p_value = stats.ttest_rel(before, after)
    is_sig = p_value < alpha
    
    return TestResult(
        test_name="Paired t-test",
        statistic=stat,
        p_value=p_value,
        is_significant=is_sig,
        interpretation=f"{'Significant' if is_sig else 'No significant'} difference at corresponding time points"
    )


def wilcoxon_test(before: np.ndarray, after: np.ndarray, alpha: float = 0.05) -> TestResult:
    """
    Perform Wilcoxon signed-rank test (non-parametric alternative to paired t-test).
    
    Parameters:
    -----------
    before : np.ndarray
        Before improvement data
    after : np.ndarray
        After improvement data
    alpha : float
        Significance level
        
    Returns:
    --------
    TestResult
        Test results with interpretation
    """
    stat, p_value = stats.wilcoxon(before, after)
    is_sig = p_value < alpha
    
    return TestResult(
        test_name="Wilcoxon Signed-Rank Test",
        statistic=stat,
        p_value=p_value,
        is_significant=is_sig,
        interpretation=f"{'Significant' if is_sig else 'No significant'} difference (non-parametric)"
    )


def independent_t_test(before: np.ndarray, after: np.ndarray, alpha: float = 0.05) -> TestResult:
    """
    Perform independent two-sample t-test.
    
    Parameters:
    -----------
    before : np.ndarray
        Before improvement data
    after : np.ndarray
        After improvement data
    alpha : float
        Significance level
        
    Returns:
    --------
    TestResult
        Test results with interpretation
    """
    stat, p_value = stats.ttest_ind(before, after)
    is_sig = p_value < alpha
    
    return TestResult(
        test_name="Independent Two-Sample t-test",
        statistic=stat,
        p_value=p_value,
        is_significant=is_sig,
        interpretation=f"{'Significant' if is_sig else 'No significant'} difference between groups"
    )


def mann_whitney_test(before: np.ndarray, after: np.ndarray, alpha: float = 0.05) -> TestResult:
    """
    Perform Mann-Whitney U test (non-parametric, one-tailed: Before > After).
    
    Parameters:
    -----------
    before : np.ndarray
        Before improvement data
    after : np.ndarray
        After improvement data
    alpha : float
        Significance level
        
    Returns:
    --------
    TestResult
        Test results with interpretation
    """
    stat, p_value = stats.mannwhitneyu(before, after, alternative='greater')
    is_sig = p_value < alpha
    
    return TestResult(
        test_name="Mann-Whitney U Test (one-tailed)",
        statistic=stat,
        p_value=p_value,
        is_significant=is_sig,
        interpretation=f"Before {'is' if is_sig else 'is not'} significantly greater than After"
    )


def levene_test(before: np.ndarray, after: np.ndarray, alpha: float = 0.05) -> TestResult:
    """
    Perform Levene's test for equality of variances.
    
    Parameters:
    -----------
    before : np.ndarray
        Before improvement data
    after : np.ndarray
        After improvement data
    alpha : float
        Significance level
        
    Returns:
    --------
    TestResult
        Test results with interpretation
    """
    stat, p_value = stats.levene(before, after)
    variances_equal = p_value > alpha
    
    return TestResult(
        test_name="Levene's Test (Equality of Variances)",
        statistic=stat,
        p_value=p_value,
        is_significant=not variances_equal,
        interpretation=f"Variances are {'equal' if variances_equal else 'different'}"
    )


def calculate_effect_size(before: np.ndarray, after: np.ndarray) -> EffectSize:
    """
    Calculate effect size metrics (Cohen's d and Glass's delta).
    
    Parameters:
    -----------
    before : np.ndarray
        Before improvement data
    after : np.ndarray
        After improvement data
        
    Returns:
    --------
    EffectSize
        Effect size metrics with interpretation
    """
    n = len(before)
    
    # Pooled standard deviation for Cohen's d
    pooled_std = np.sqrt(
        ((n - 1) * np.std(before, ddof=1)**2 + (n - 1) * np.std(after, ddof=1)**2) / 
        (2 * n - 2)
    )
    
    cohens_d = (np.mean(before) - np.mean(after)) / pooled_std
    glass_delta = (np.mean(before) - np.mean(after)) / np.std(before, ddof=1)
    
    # Interpretation based on Cohen's guidelines
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        interpretation = "Negligible"
    elif abs_d < 0.5:
        interpretation = "Small"
    elif abs_d < 0.8:
        interpretation = "Medium"
    else:
        interpretation = "Large"
    
    return EffectSize(
        cohens_d=cohens_d,
        glass_delta=glass_delta,
        interpretation=interpretation
    )


def calculate_confidence_interval(
    before: np.ndarray, 
    after: np.ndarray, 
    confidence: float = 0.95
) -> ConfidenceInterval:
    """
    Calculate confidence interval for the mean difference.
    
    Parameters:
    -----------
    before : np.ndarray
        Before improvement data
    after : np.ndarray
        After improvement data
    confidence : float
        Confidence level (default 0.95 for 95% CI)
        
    Returns:
    --------
    ConfidenceInterval
        Confidence interval details
    """
    diff = before - after
    mean_diff = np.mean(diff)
    std_error = stats.sem(diff)
    
    ci = stats.t.interval(confidence, len(diff) - 1, loc=mean_diff, scale=std_error)
    
    return ConfidenceInterval(
        mean_diff=mean_diff,
        std_error=std_error,
        lower=ci[0],
        upper=ci[1],
        confidence_level=confidence
    )


def analyze_by_time_period(
    before: np.ndarray,
    after: np.ndarray,
    periods: Dict[str, Tuple[int, int]] = None,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Analyze improvement by time periods.
    
    Parameters:
    -----------
    before : np.ndarray
        Before improvement data
    after : np.ndarray
        After improvement data
    periods : Dict[str, Tuple[int, int]]
        Dictionary mapping period names to (start_index, end_index)
        Default assumes 96 data points (15-min intervals for 24 hours)
    alpha : float
        Significance level
        
    Returns:
    --------
    pd.DataFrame
        Analysis results by time period
    """
    if periods is None:
        # Default: 24-hour day split into 4 periods (96 points = 15-min intervals)
        periods = {
            'Night (00:00-06:00)': (0, 24),
            'Morning (06:00-12:00)': (24, 48),
            'Afternoon (12:00-18:00)': (48, 72),
            'Evening (18:00-24:00)': (72, 96)
        }
    
    results = []
    for period_name, (start, end) in periods.items():
        before_period = before[start:end]
        after_period = after[start:end]
        
        _, p_val = stats.ttest_rel(before_period, after_period)
        
        results.append({
            'Period': period_name,
            'Before Mean': np.mean(before_period),
            'After Mean': np.mean(after_period),
            'Reduction': np.mean(before_period) - np.mean(after_period),
            'Reduction %': ((np.mean(before_period) - np.mean(after_period)) / 
                          np.mean(before_period)) * 100,
            'p-value': p_val,
            'Significant': p_val < alpha
        })
    
    return pd.DataFrame(results)


def analyze_improvement(
    before: np.ndarray,
    after: np.ndarray,
    alpha: float = 0.05,
    confidence: float = 0.95,
    periods: Dict[str, Tuple[int, int]] = None
) -> AnalysisResults:
    """
    Perform comprehensive statistical analysis to determine if there was improvement.
    
    Parameters:
    -----------
    before : np.ndarray
        CPU usage data before improvement (e.g., 96 measurements)
    after : np.ndarray
        CPU usage data after improvement (same length as before)
    alpha : float
        Significance level for statistical tests (default 0.05)
    confidence : float
        Confidence level for CI (default 0.95)
    periods : Dict[str, Tuple[int, int]]
        Optional time period definitions for period analysis
        
    Returns:
    --------
    AnalysisResults
        Complete analysis results including all tests and metrics
    """
    # Validate inputs
    if len(before) != len(after):
        raise ValueError("Before and after arrays must have the same length")
    
    # Descriptive statistics
    before_stats = calculate_descriptive_stats(before)
    after_stats = calculate_descriptive_stats(after)
    
    # Calculate improvement percentage
    improvement_pct = ((before_stats.mean - after_stats.mean) / before_stats.mean) * 100
    
    # Run all statistical tests
    tests = [
        test_normality(before, alpha),
        test_normality(after, alpha),
        paired_t_test(before, after, alpha),
        wilcoxon_test(before, after, alpha),
        independent_t_test(before, after, alpha),
        mann_whitney_test(before, after, alpha),
        levene_test(before, after, alpha)
    ]
    
    # Effect size
    effect_size = calculate_effect_size(before, after)
    
    # Confidence interval
    ci = calculate_confidence_interval(before, after, confidence)
    
    # Period analysis (if data length matches expected)
    period_analysis = None
    if len(before) == 96:  # Standard 24-hour, 15-min interval data
        period_analysis = analyze_by_time_period(before, after, periods, alpha)
    
    # Determine verdict
    paired_test = tests[2]  # Paired t-test
    wilcoxon = tests[3]  # Wilcoxon test
    
    if paired_test.is_significant and effect_size.cohens_d > 0.2:
        verdict = "YES - Significant Improvement Detected"
        verdict_details = (
            f"The analysis shows a statistically significant improvement "
            f"(p < {alpha}) with a {effect_size.interpretation.lower()} effect size "
            f"(Cohen's d = {effect_size.cohens_d:.2f}). "
            f"Mean CPU usage decreased by {improvement_pct:.1f}%."
        )
    elif paired_test.is_significant:
        verdict = "MARGINAL - Statistically Significant but Small Effect"
        verdict_details = (
            f"The improvement is statistically significant (p < {alpha}) "
            f"but the effect size is {effect_size.interpretation.lower()} "
            f"(Cohen's d = {effect_size.cohens_d:.2f}). "
            f"The practical significance may be limited."
        )
    else:
        verdict = "NO - No Significant Improvement"
        verdict_details = (
            f"The analysis does not show a statistically significant improvement "
            f"(p = {paired_test.p_value:.4f} > {alpha}). "
            f"Any observed difference could be due to random variation."
        )
    
    return AnalysisResults(
        before_stats=before_stats,
        after_stats=after_stats,
        improvement_pct=improvement_pct,
        tests=tests,
        effect_size=effect_size,
        confidence_interval=ci,
        period_analysis=period_analysis,
        verdict=verdict,
        verdict_details=verdict_details
    )


def print_analysis_report(results: AnalysisResults) -> str:
    """
    Generate a formatted text report from analysis results.
    
    Parameters:
    -----------
    results : AnalysisResults
        Results from analyze_improvement()
        
    Returns:
    --------
    str
        Formatted report string
    """
    report = []
    report.append("=" * 70)
    report.append("CPU USAGE IMPROVEMENT ANALYSIS REPORT")
    report.append("=" * 70)
    
    # Descriptive Statistics
    report.append("\n1. DESCRIPTIVE STATISTICS")
    report.append("-" * 50)
    report.append(f"{'Metric':<15} {'Before':>12} {'After':>12} {'Difference':>12}")
    report.append("-" * 50)
    
    metrics = [
        ('Mean', results.before_stats.mean, results.after_stats.mean),
        ('Median', results.before_stats.median, results.after_stats.median),
        ('Std Dev', results.before_stats.std, results.after_stats.std),
        ('Min', results.before_stats.min, results.after_stats.min),
        ('Max', results.before_stats.max, results.after_stats.max),
        ('IQR', results.before_stats.iqr, results.after_stats.iqr),
        ('CV (%)', results.before_stats.cv, results.after_stats.cv),
    ]
    
    for name, before, after in metrics:
        report.append(f"{name:<15} {before:>12.2f} {after:>12.2f} {before - after:>12.2f}")
    
    report.append(f"\n>> Mean CPU Reduction: {results.improvement_pct:.2f}%")
    
    # Statistical Tests
    report.append("\n" + "=" * 70)
    report.append("\n2. STATISTICAL TESTS")
    report.append("-" * 50)
    
    for test in results.tests:
        report.append(f"\n{test.test_name}")
        report.append(f"  Statistic: {test.statistic:.4f}")
        report.append(f"  p-value: {test.p_value:.6f}")
        report.append(f"  Result: {test.interpretation}")
    
    # Effect Size
    report.append("\n" + "=" * 70)
    report.append("\n3. EFFECT SIZE")
    report.append("-" * 50)
    report.append(f"Cohen's d: {results.effect_size.cohens_d:.4f}")
    report.append(f"Glass's Delta: {results.effect_size.glass_delta:.4f}")
    report.append(f"Interpretation: {results.effect_size.interpretation} effect")
    
    # Confidence Interval
    report.append("\n" + "=" * 70)
    report.append("\n4. CONFIDENCE INTERVAL")
    report.append("-" * 50)
    ci = results.confidence_interval
    report.append(f"Mean Difference: {ci.mean_diff:.2f}%")
    report.append(f"Standard Error: {ci.std_error:.2f}")
    report.append(f"{ci.confidence_level*100:.0f}% CI: [{ci.lower:.2f}, {ci.upper:.2f}]")
    
    # Period Analysis
    if results.period_analysis is not None:
        report.append("\n" + "=" * 70)
        report.append("\n5. TIME PERIOD ANALYSIS")
        report.append("-" * 50)
        for _, row in results.period_analysis.iterrows():
            sig = "*" if row['Significant'] else ""
            report.append(
                f"{row['Period']:<25} "
                f"Before: {row['Before Mean']:>6.2f}  "
                f"After: {row['After Mean']:>6.2f}  "
                f"Reduction: {row['Reduction']:>6.2f} ({row['Reduction %']:>5.1f}%)  "
                f"p={row['p-value']:.4f}{sig}"
            )
        report.append("\n* indicates statistically significant (p < 0.05)")
    
    # Verdict
    report.append("\n" + "=" * 70)
    report.append("\n6. VERDICT")
    report.append("=" * 70)
    report.append(f"\n{results.verdict}")
    report.append(f"\n{results.verdict_details}")
    
    return "\n".join(report)


if __name__ == "__main__":
    # Example usage with random data
    np.random.seed(42)
    n = 96
    
    # Generate sample data
    before = np.random.normal(60, 15, n)
    after = np.random.normal(45, 12, n)
    
    # Run analysis
    results = analyze_improvement(before, after)
    
    # Print report
    print(print_analysis_report(results))
