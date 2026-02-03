"""
Multi-Day CPU Usage Analysis
============================
Statistical analysis approaches when you have multiple days of data
before and after a change.

Approaches:
1. Concatenate: Treat all data points as one series
2. Average: Average across days to get "typical day" pattern
3. Day-paired: Compare corresponding days
4. Mixed Effects: Account for day-to-day variation
5. Bootstrap: Resample-based confidence intervals

Author: Statistical Analysis Tool
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import warnings


@dataclass
class MultiDayResults:
    """Results container for multi-day analysis"""
    approach: str
    description: str
    before_mean: float
    after_mean: float
    improvement_pct: float
    statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    is_significant: bool
    additional_info: Dict


def generate_multiday_data(
    n_days_before: int = 2,
    n_days_after: int = 2,
    points_per_day: int = 96,
    improvement_pct: float = 0.25,
    day_to_day_variation: float = 5.0,
    random_seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Generate synthetic multi-day CPU usage data.
    
    Parameters:
    -----------
    n_days_before : int
        Number of days of data before the change
    n_days_after : int
        Number of days of data after the change
    points_per_day : int
        Measurements per day (96 = 15-min intervals)
    improvement_pct : float
        Expected improvement as fraction (0.25 = 25%)
    day_to_day_variation : float
        Standard deviation of day-to-day baseline shifts
    random_seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    before_data : np.ndarray
        Shape (n_days_before, points_per_day)
    after_data : np.ndarray
        Shape (n_days_after, points_per_day)
    df : pd.DataFrame
        Long-format dataframe with all data
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Base daily pattern
    def daily_pattern(n_points):
        pattern = np.zeros(n_points)
        points_per_hour = n_points / 24
        for i in range(n_points):
            hour = i / points_per_hour
            if 0 <= hour < 6:
                pattern[i] = 30
            elif 6 <= hour < 9:
                pattern[i] = 30 + (hour - 6) * 10
            elif 9 <= hour < 18:
                pattern[i] = 60 + 15 * np.sin((hour - 9) * np.pi / 9)
            elif 18 <= hour < 22:
                pattern[i] = 60 - (hour - 18) * 8
            else:
                pattern[i] = 28
        return pattern
    
    base_pattern = daily_pattern(points_per_day)
    
    # Generate BEFORE data (multiple days)
    before_data = np.zeros((n_days_before, points_per_day))
    for day in range(n_days_before):
        day_shift = np.random.normal(0, day_to_day_variation)  # Day-to-day variation
        noise = np.random.normal(0, 8, points_per_day)
        spikes = np.random.choice([0, 15, 25], points_per_day, p=[0.85, 0.10, 0.05])
        before_data[day] = base_pattern + day_shift + noise + spikes
    
    before_data = np.clip(before_data, 5, 98)
    
    # Generate AFTER data (multiple days) with improvement
    after_data = np.zeros((n_days_after, points_per_day))
    for day in range(n_days_after):
        day_shift = np.random.normal(0, day_to_day_variation * 0.8)  # Slightly more stable
        noise = np.random.normal(0, 6, points_per_day)  # Less noise
        spikes = np.random.choice([0, 8, 12], points_per_day, p=[0.92, 0.06, 0.02])
        after_data[day] = base_pattern * (1 - improvement_pct) + day_shift + noise + spikes
    
    after_data = np.clip(after_data, 5, 95)
    
    # Create long-format DataFrame
    records = []
    for day in range(n_days_before):
        for point in range(points_per_day):
            hour = (point * 15) // 60
            minute = (point * 15) % 60
            records.append({
                'Condition': 'Before',
                'Day': day + 1,
                'Time_Index': point,
                'Time': f'{hour:02d}:{minute:02d}',
                'CPU_Usage': before_data[day, point]
            })
    
    for day in range(n_days_after):
        for point in range(points_per_day):
            hour = (point * 15) // 60
            minute = (point * 15) % 60
            records.append({
                'Condition': 'After',
                'Day': day + 1,
                'Time_Index': point,
                'Time': f'{hour:02d}:{minute:02d}',
                'CPU_Usage': after_data[day, point]
            })
    
    df = pd.DataFrame(records)
    
    return before_data, after_data, df


# =============================================================================
# APPROACH 1: CONCATENATE ALL DATA
# =============================================================================

def analyze_concatenated(
    before_data: np.ndarray,
    after_data: np.ndarray,
    alpha: float = 0.05
) -> MultiDayResults:
    """
    Approach 1: Concatenate all days into single series.
    
    Treats each measurement as independent observation.
    Simple but ignores day-to-day structure.
    
    Parameters:
    -----------
    before_data : np.ndarray
        Shape (n_days, points_per_day)
    after_data : np.ndarray
        Shape (n_days, points_per_day)
    alpha : float
        Significance level
        
    Returns:
    --------
    MultiDayResults
    """
    before_flat = before_data.flatten()
    after_flat = after_data.flatten()
    
    # Independent samples t-test
    t_stat, p_value = stats.ttest_ind(before_flat, after_flat)
    
    # Mann-Whitney U (non-parametric)
    u_stat, p_mann = stats.mannwhitneyu(before_flat, after_flat, alternative='greater')
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(
        ((len(before_flat) - 1) * np.std(before_flat, ddof=1)**2 +
         (len(after_flat) - 1) * np.std(after_flat, ddof=1)**2) /
        (len(before_flat) + len(after_flat) - 2)
    )
    cohens_d = (np.mean(before_flat) - np.mean(after_flat)) / pooled_std
    
    # Confidence interval for difference
    diff_mean = np.mean(before_flat) - np.mean(after_flat)
    se = np.sqrt(np.var(before_flat, ddof=1)/len(before_flat) + 
                 np.var(after_flat, ddof=1)/len(after_flat))
    ci = (diff_mean - 1.96*se, diff_mean + 1.96*se)
    
    improvement = (np.mean(before_flat) - np.mean(after_flat)) / np.mean(before_flat) * 100
    
    return MultiDayResults(
        approach="Concatenated",
        description="All measurements pooled into single series (ignores day structure)",
        before_mean=np.mean(before_flat),
        after_mean=np.mean(after_flat),
        improvement_pct=improvement,
        statistic=t_stat,
        p_value=p_value,
        effect_size=cohens_d,
        confidence_interval=ci,
        is_significant=p_value < alpha,
        additional_info={
            'n_before': len(before_flat),
            'n_after': len(after_flat),
            'mann_whitney_p': p_mann
        }
    )


# =============================================================================
# APPROACH 2: AVERAGE ACROSS DAYS (Typical Day)
# =============================================================================

def analyze_averaged(
    before_data: np.ndarray,
    after_data: np.ndarray,
    alpha: float = 0.05
) -> MultiDayResults:
    """
    Approach 2: Average across days to create "typical day" profile.
    
    Reduces noise by averaging, then compares the typical patterns.
    Good for identifying consistent time-of-day effects.
    
    Parameters:
    -----------
    before_data : np.ndarray
        Shape (n_days, points_per_day)
    after_data : np.ndarray
        Shape (n_days, points_per_day)
    alpha : float
        Significance level
        
    Returns:
    --------
    MultiDayResults
    """
    # Average across days
    before_avg = np.mean(before_data, axis=0)
    after_avg = np.mean(after_data, axis=0)
    
    # Standard error at each time point
    before_se = stats.sem(before_data, axis=0)
    after_se = stats.sem(after_data, axis=0)
    
    # Paired t-test on averaged profiles
    t_stat, p_value = stats.ttest_rel(before_avg, after_avg)
    
    # Wilcoxon (non-parametric)
    w_stat, p_wilcox = stats.wilcoxon(before_avg, after_avg)
    
    # Effect size
    diff = before_avg - after_avg
    cohens_d = np.mean(diff) / np.std(diff, ddof=1)
    
    # CI for mean difference
    mean_diff = np.mean(diff)
    se_diff = stats.sem(diff)
    ci = stats.t.interval(0.95, len(diff)-1, loc=mean_diff, scale=se_diff)
    
    improvement = (np.mean(before_avg) - np.mean(after_avg)) / np.mean(before_avg) * 100
    
    return MultiDayResults(
        approach="Averaged (Typical Day)",
        description="Days averaged to create typical daily profile, then compared",
        before_mean=np.mean(before_avg),
        after_mean=np.mean(after_avg),
        improvement_pct=improvement,
        statistic=t_stat,
        p_value=p_value,
        effect_size=cohens_d,
        confidence_interval=ci,
        is_significant=p_value < alpha,
        additional_info={
            'n_timepoints': len(before_avg),
            'n_days_before': before_data.shape[0],
            'n_days_after': after_data.shape[0],
            'wilcoxon_p': p_wilcox,
            'before_day_variability': np.mean(np.std(before_data, axis=0)),
            'after_day_variability': np.mean(np.std(after_data, axis=0))
        }
    )


# =============================================================================
# APPROACH 3: DAY-LEVEL ANALYSIS (Compare Daily Means)
# =============================================================================

def analyze_daily_means(
    before_data: np.ndarray,
    after_data: np.ndarray,
    alpha: float = 0.05
) -> MultiDayResults:
    """
    Approach 3: Compare daily means.
    
    Each day becomes one observation. Good when day-to-day variation
    is the primary source of variability.
    
    Parameters:
    -----------
    before_data : np.ndarray
        Shape (n_days, points_per_day)
    after_data : np.ndarray
        Shape (n_days, points_per_day)
    alpha : float
        Significance level
        
    Returns:
    --------
    MultiDayResults
    """
    # Calculate daily means
    before_daily = np.mean(before_data, axis=1)
    after_daily = np.mean(after_data, axis=1)
    
    n_before = len(before_daily)
    n_after = len(after_daily)
    
    # Independent samples t-test (or Welch's t-test)
    t_stat, p_value = stats.ttest_ind(before_daily, after_daily, equal_var=False)
    
    # Mann-Whitney (for small samples)
    if n_before >= 3 and n_after >= 3:
        u_stat, p_mann = stats.mannwhitneyu(before_daily, after_daily, alternative='greater')
    else:
        p_mann = np.nan
    
    # Effect size
    pooled_std = np.sqrt(
        ((n_before - 1) * np.std(before_daily, ddof=1)**2 +
         (n_after - 1) * np.std(after_daily, ddof=1)**2) /
        (n_before + n_after - 2)
    )
    cohens_d = (np.mean(before_daily) - np.mean(after_daily)) / pooled_std if pooled_std > 0 else 0
    
    # CI for difference
    diff_mean = np.mean(before_daily) - np.mean(after_daily)
    se = np.sqrt(np.var(before_daily, ddof=1)/n_before + 
                 np.var(after_daily, ddof=1)/n_after)
    df = n_before + n_after - 2
    ci = stats.t.interval(0.95, df, loc=diff_mean, scale=se)
    
    improvement = diff_mean / np.mean(before_daily) * 100
    
    return MultiDayResults(
        approach="Daily Means",
        description="Each day is one observation; compares mean of daily averages",
        before_mean=np.mean(before_daily),
        after_mean=np.mean(after_daily),
        improvement_pct=improvement,
        statistic=t_stat,
        p_value=p_value,
        effect_size=cohens_d,
        confidence_interval=ci,
        is_significant=p_value < alpha,
        additional_info={
            'n_days_before': n_before,
            'n_days_after': n_after,
            'before_daily_means': before_daily.tolist(),
            'after_daily_means': after_daily.tolist(),
            'mann_whitney_p': p_mann,
            'warning': 'Low power with few days' if (n_before < 5 or n_after < 5) else None
        }
    )


# =============================================================================
# APPROACH 4: MIXED EFFECTS / HIERARCHICAL MODEL
# =============================================================================

def analyze_mixed_effects(
    before_data: np.ndarray,
    after_data: np.ndarray,
    alpha: float = 0.05
) -> MultiDayResults:
    """
    Approach 4: Mixed effects model approximation.
    
    Accounts for both within-day and between-day variation.
    Uses a simplified approach without requiring statsmodels.
    
    The model conceptually is:
    CPU ~ Condition + (1|Day) + (1|TimePoint)
    
    Parameters:
    -----------
    before_data : np.ndarray
        Shape (n_days, points_per_day)
    after_data : np.ndarray
        Shape (n_days, points_per_day)
    alpha : float
        Significance level
        
    Returns:
    --------
    MultiDayResults
    """
    n_days_before, n_points = before_data.shape
    n_days_after = after_data.shape[0]
    
    # Calculate variance components
    # Between-day variance (before)
    day_means_before = np.mean(before_data, axis=1)
    var_between_before = np.var(day_means_before, ddof=1) if n_days_before > 1 else 0
    
    # Within-day variance (before)
    var_within_before = np.mean([np.var(before_data[d], ddof=1) for d in range(n_days_before)])
    
    # Between-day variance (after)
    day_means_after = np.mean(after_data, axis=1)
    var_between_after = np.var(day_means_after, ddof=1) if n_days_after > 1 else 0
    
    # Within-day variance (after)
    var_within_after = np.mean([np.var(after_data[d], ddof=1) for d in range(n_days_after)])
    
    # Effective sample size accounting for clustering
    # ICC (Intraclass Correlation Coefficient)
    total_var_before = var_between_before + var_within_before
    icc_before = var_between_before / total_var_before if total_var_before > 0 else 0
    
    total_var_after = var_between_after + var_within_after
    icc_after = var_between_after / total_var_after if total_var_after > 0 else 0
    
    # Design effect
    design_effect_before = 1 + (n_points - 1) * icc_before
    design_effect_after = 1 + (n_points - 1) * icc_after
    
    # Effective sample sizes
    n_eff_before = (n_days_before * n_points) / design_effect_before
    n_eff_after = (n_days_after * n_points) / design_effect_after
    
    # Adjusted standard errors
    before_flat = before_data.flatten()
    after_flat = after_data.flatten()
    
    se_before = np.std(before_flat, ddof=1) / np.sqrt(n_eff_before)
    se_after = np.std(after_flat, ddof=1) / np.sqrt(n_eff_after)
    
    # Test statistic with adjusted SE
    diff_mean = np.mean(before_flat) - np.mean(after_flat)
    se_diff = np.sqrt(se_before**2 + se_after**2)
    t_stat = diff_mean / se_diff
    
    # Approximate degrees of freedom (Satterthwaite)
    df = (se_diff**4) / (se_before**4/(n_eff_before-1) + se_after**4/(n_eff_after-1))
    
    # P-value
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    
    # Effect size
    pooled_std = np.sqrt((var_between_before + var_within_before + 
                          var_between_after + var_within_after) / 2)
    cohens_d = diff_mean / pooled_std if pooled_std > 0 else 0
    
    # CI
    ci = (diff_mean - stats.t.ppf(0.975, df) * se_diff,
          diff_mean + stats.t.ppf(0.975, df) * se_diff)
    
    improvement = diff_mean / np.mean(before_flat) * 100
    
    return MultiDayResults(
        approach="Mixed Effects (Approximation)",
        description="Accounts for clustering within days using design effect adjustment",
        before_mean=np.mean(before_flat),
        after_mean=np.mean(after_flat),
        improvement_pct=improvement,
        statistic=t_stat,
        p_value=p_value,
        effect_size=cohens_d,
        confidence_interval=ci,
        is_significant=p_value < alpha,
        additional_info={
            'icc_before': icc_before,
            'icc_after': icc_after,
            'design_effect_before': design_effect_before,
            'design_effect_after': design_effect_after,
            'effective_n_before': n_eff_before,
            'effective_n_after': n_eff_after,
            'var_between_before': var_between_before,
            'var_within_before': var_within_before,
            'var_between_after': var_between_after,
            'var_within_after': var_within_after,
            'approx_df': df
        }
    )


# =============================================================================
# APPROACH 5: BOOTSTRAP CONFIDENCE INTERVAL
# =============================================================================

def analyze_bootstrap(
    before_data: np.ndarray,
    after_data: np.ndarray,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    random_seed: Optional[int] = None
) -> MultiDayResults:
    """
    Approach 5: Bootstrap resampling for robust inference.
    
    Resamples days (cluster bootstrap) to get confidence intervals
    that account for within-day correlation.
    
    Parameters:
    -----------
    before_data : np.ndarray
        Shape (n_days, points_per_day)
    after_data : np.ndarray
        Shape (n_days, points_per_day)
    n_bootstrap : int
        Number of bootstrap iterations
    alpha : float
        Significance level
    random_seed : int, optional
        Random seed
        
    Returns:
    --------
    MultiDayResults
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n_days_before = before_data.shape[0]
    n_days_after = after_data.shape[0]
    
    observed_diff = np.mean(before_data) - np.mean(after_data)
    
    # Cluster bootstrap (resample days, not individual points)
    bootstrap_diffs = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        # Resample days with replacement
        before_idx = np.random.choice(n_days_before, n_days_before, replace=True)
        after_idx = np.random.choice(n_days_after, n_days_after, replace=True)
        
        before_sample = before_data[before_idx].flatten()
        after_sample = after_data[after_idx].flatten()
        
        bootstrap_diffs[i] = np.mean(before_sample) - np.mean(after_sample)
    
    # Bootstrap confidence interval (percentile method)
    ci_lower = np.percentile(bootstrap_diffs, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))
    
    # Bootstrap p-value (proportion of samples crossing zero)
    # For one-tailed test (before > after)
    p_value = np.mean(bootstrap_diffs <= 0)
    
    # Two-tailed p-value
    p_value_two_tailed = 2 * min(p_value, 1 - p_value)
    
    # Effect size
    before_flat = before_data.flatten()
    after_flat = after_data.flatten()
    pooled_std = np.sqrt((np.var(before_flat, ddof=1) + np.var(after_flat, ddof=1)) / 2)
    cohens_d = observed_diff / pooled_std if pooled_std > 0 else 0
    
    improvement = observed_diff / np.mean(before_flat) * 100
    
    return MultiDayResults(
        approach="Bootstrap (Cluster)",
        description="Cluster bootstrap resampling days to account for within-day correlation",
        before_mean=np.mean(before_flat),
        after_mean=np.mean(after_flat),
        improvement_pct=improvement,
        statistic=observed_diff,  # Using observed difference as "statistic"
        p_value=p_value_two_tailed,
        effect_size=cohens_d,
        confidence_interval=(ci_lower, ci_upper),
        is_significant=0 < ci_lower or ci_upper < 0,  # CI doesn't include 0
        additional_info={
            'n_bootstrap': n_bootstrap,
            'bootstrap_mean': np.mean(bootstrap_diffs),
            'bootstrap_std': np.std(bootstrap_diffs),
            'one_tailed_p': p_value,
            'ci_method': 'percentile'
        }
    )


# =============================================================================
# COMPREHENSIVE ANALYSIS
# =============================================================================

def run_all_approaches(
    before_data: np.ndarray,
    after_data: np.ndarray,
    alpha: float = 0.05,
    bootstrap_seed: Optional[int] = None
) -> List[MultiDayResults]:
    """
    Run all analysis approaches and return results.
    
    Parameters:
    -----------
    before_data : np.ndarray
        Shape (n_days, points_per_day)
    after_data : np.ndarray
        Shape (n_days, points_per_day)
    alpha : float
        Significance level
    bootstrap_seed : int, optional
        Seed for bootstrap reproducibility
        
    Returns:
    --------
    List[MultiDayResults]
    """
    results = []
    
    results.append(analyze_concatenated(before_data, after_data, alpha))
    results.append(analyze_averaged(before_data, after_data, alpha))
    results.append(analyze_daily_means(before_data, after_data, alpha))
    results.append(analyze_mixed_effects(before_data, after_data, alpha))
    results.append(analyze_bootstrap(before_data, after_data, alpha=alpha, 
                                     random_seed=bootstrap_seed))
    
    return results


def print_multiday_report(results: List[MultiDayResults]) -> str:
    """
    Generate a formatted report comparing all approaches.
    
    Parameters:
    -----------
    results : List[MultiDayResults]
        Results from run_all_approaches()
        
    Returns:
    --------
    str
        Formatted report
    """
    report = []
    report.append("=" * 80)
    report.append("MULTI-DAY CPU USAGE ANALYSIS - COMPARISON OF APPROACHES")
    report.append("=" * 80)
    
    # Summary table
    report.append("\n" + "-" * 80)
    report.append(f"{'Approach':<30} {'Before':>10} {'After':>10} {'Improve%':>10} {'p-value':>12} {'Sig?':>6}")
    report.append("-" * 80)
    
    for r in results:
        sig = "YES" if r.is_significant else "no"
        report.append(f"{r.approach:<30} {r.before_mean:>10.2f} {r.after_mean:>10.2f} "
                     f"{r.improvement_pct:>10.1f} {r.p_value:>12.6f} {sig:>6}")
    
    report.append("-" * 80)
    
    # Detailed results
    report.append("\n" + "=" * 80)
    report.append("DETAILED RESULTS BY APPROACH")
    report.append("=" * 80)
    
    for r in results:
        report.append(f"\n{'─' * 60}")
        report.append(f"APPROACH: {r.approach}")
        report.append(f"{'─' * 60}")
        report.append(f"Description: {r.description}")
        report.append(f"\nKey Statistics:")
        report.append(f"  Before Mean: {r.before_mean:.2f}%")
        report.append(f"  After Mean:  {r.after_mean:.2f}%")
        report.append(f"  Improvement: {r.improvement_pct:.1f}%")
        report.append(f"  Test Statistic: {r.statistic:.4f}")
        report.append(f"  P-value: {r.p_value:.6f}")
        report.append(f"  Effect Size (Cohen's d): {r.effect_size:.3f}")
        report.append(f"  95% CI: [{r.confidence_interval[0]:.2f}, {r.confidence_interval[1]:.2f}]")
        report.append(f"  Significant (α=0.05): {'YES' if r.is_significant else 'NO'}")
        
        if r.additional_info:
            report.append(f"\nAdditional Info:")
            for key, value in r.additional_info.items():
                if value is not None:
                    if isinstance(value, float):
                        report.append(f"  {key}: {value:.4f}")
                    else:
                        report.append(f"  {key}: {value}")
    
    # Recommendations
    report.append("\n" + "=" * 80)
    report.append("RECOMMENDATIONS")
    report.append("=" * 80)
    
    report.append("""
Which approach to use:

1. CONCATENATED: Use when you have many days and measurements are 
   relatively independent. Simple but may inflate significance.

2. AVERAGED (Typical Day): Best for understanding time-of-day patterns.
   Use when daily patterns are consistent and you care about 
   comparing typical profiles.

3. DAILY MEANS: Most conservative. Each day is one observation.
   Use when day-to-day variation is high and you have 5+ days.
   ⚠️ Low power with only 2 days!

4. MIXED EFFECTS: Best balance of power and validity. Accounts for
   both within-day and between-day variation. Recommended for most cases.

5. BOOTSTRAP: Most robust. Makes no distributional assumptions.
   Good when you're unsure about data properties. Requires adequate
   number of days for reliable cluster bootstrap (ideally 5+).

For 2 days before/after:
→ AVERAGED or MIXED EFFECTS approaches are recommended
→ DAILY MEANS has very low power (only 4 observations total!)
→ Consider collecting more days if possible
""")
    
    # Consensus
    n_significant = sum(1 for r in results if r.is_significant)
    report.append(f"\nCONSENSUS: {n_significant}/{len(results)} approaches show significant improvement")
    
    if n_significant >= 4:
        report.append("VERDICT: Strong evidence of improvement across multiple analytical approaches")
    elif n_significant >= 2:
        report.append("VERDICT: Moderate evidence of improvement (results vary by approach)")
    else:
        report.append("VERDICT: Weak or no evidence of improvement")
    
    return "\n".join(report)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("Generating synthetic multi-day data (2 days before, 2 days after)...")
    
    before_data, after_data, df = generate_multiday_data(
        n_days_before=2,
        n_days_after=2,
        points_per_day=96,
        improvement_pct=0.25,
        random_seed=42
    )
    
    print(f"Data shape - Before: {before_data.shape}, After: {after_data.shape}")
    print(f"Total measurements: {before_data.size + after_data.size}")
    
    # Run all approaches
    results = run_all_approaches(before_data, after_data, bootstrap_seed=42)
    
    # Print report
    report = print_multiday_report(results)
    print(report)
    
    # Save data
    df.to_csv('multiday_cpu_data.csv', index=False)
    print("\nData saved to: multiday_cpu_data.csv")
