"""
Synthetic Data Generator for CPU Usage Analysis
================================================
This module generates realistic synthetic CPU usage data
for testing the statistical analysis and visualization modules.

Author: Statistical Analysis Tool
Usage: Import and call generate_cpu_data()
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class DataGenerationConfig:
    """Configuration for synthetic data generation"""
    n_points: int = 96  # Default: 15-min intervals for 24 hours
    base_cpu_before: float = 50.0  # Base CPU level before improvement
    base_cpu_after: float = 35.0   # Base CPU level after improvement
    noise_std_before: float = 8.0  # Noise standard deviation before
    noise_std_after: float = 5.0   # Noise standard deviation after
    spike_probability_before: float = 0.15  # Probability of CPU spikes before
    spike_probability_after: float = 0.05   # Probability of CPU spikes after
    spike_magnitude_before: float = 20.0    # Spike magnitude before
    spike_magnitude_after: float = 10.0     # Spike magnitude after
    daily_pattern_amplitude: float = 20.0   # Amplitude of daily pattern
    random_seed: Optional[int] = None       # Random seed for reproducibility


def generate_daily_pattern(n_points: int, amplitude: float = 20.0) -> np.ndarray:
    """
    Generate a realistic daily CPU usage pattern.
    
    Simulates typical server load:
    - Low at night (00:00-06:00)
    - Ramp up in morning (06:00-09:00)
    - High during business hours (09:00-18:00)
    - Wind down in evening (18:00-24:00)
    
    Parameters:
    -----------
    n_points : int
        Number of data points
    amplitude : float
        Amplitude of the daily variation
        
    Returns:
    --------
    np.ndarray
        Daily pattern array
    """
    pattern = np.zeros(n_points)
    points_per_hour = n_points / 24
    
    for i in range(n_points):
        hour = (i / points_per_hour)
        
        if 0 <= hour < 6:
            # Night: low and stable
            pattern[i] = -amplitude * 0.8
        elif 6 <= hour < 9:
            # Morning ramp-up
            progress = (hour - 6) / 3
            pattern[i] = -amplitude * 0.8 + progress * amplitude * 1.5
        elif 9 <= hour < 18:
            # Business hours: high with midday peak
            midday_factor = np.sin((hour - 9) * np.pi / 9)
            pattern[i] = amplitude * 0.7 + midday_factor * amplitude * 0.3
        elif 18 <= hour < 22:
            # Evening wind-down
            progress = (hour - 18) / 4
            pattern[i] = amplitude * 0.7 - progress * amplitude * 1.2
        else:
            # Late night
            pattern[i] = -amplitude * 0.5
    
    return pattern


def generate_spikes(
    n_points: int,
    probability: float,
    magnitude: float
) -> np.ndarray:
    """
    Generate random CPU spikes.
    
    Parameters:
    -----------
    n_points : int
        Number of data points
    probability : float
        Probability of a spike at each point
    magnitude : float
        Base magnitude of spikes (actual spike is random up to this value)
        
    Returns:
    --------
    np.ndarray
        Array of spike values
    """
    spikes = np.zeros(n_points)
    spike_mask = np.random.random(n_points) < probability
    spike_values = np.random.uniform(magnitude * 0.5, magnitude, n_points)
    spikes[spike_mask] = spike_values[spike_mask]
    return spikes


def generate_cpu_data(
    config: Optional[DataGenerationConfig] = None,
    improvement_factor: float = 0.3
) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    """
    Generate synthetic CPU usage data for before/after comparison.
    
    Parameters:
    -----------
    config : DataGenerationConfig, optional
        Configuration for data generation. If None, uses defaults.
    improvement_factor : float
        Expected improvement as a fraction (0.3 = 30% reduction)
        
    Returns:
    --------
    Tuple containing:
        - before: np.ndarray of CPU usage before improvement
        - after: np.ndarray of CPU usage after improvement
        - time_labels: List of time strings
        - df: DataFrame with all data
    """
    if config is None:
        config = DataGenerationConfig()
    
    if config.random_seed is not None:
        np.random.seed(config.random_seed)
    
    # Generate time labels
    start_time = datetime(2024, 1, 1, 0, 0)
    interval_minutes = (24 * 60) // config.n_points
    time_intervals = [
        start_time + timedelta(minutes=interval_minutes * i) 
        for i in range(config.n_points)
    ]
    time_labels = [t.strftime('%H:%M') for t in time_intervals]
    
    # Generate daily pattern
    daily_pattern = generate_daily_pattern(config.n_points, config.daily_pattern_amplitude)
    
    # Generate BEFORE data
    before_noise = np.random.normal(0, config.noise_std_before, config.n_points)
    before_spikes = generate_spikes(
        config.n_points, 
        config.spike_probability_before, 
        config.spike_magnitude_before
    )
    before = config.base_cpu_before + daily_pattern + before_noise + before_spikes
    before = np.clip(before, 5, 98)  # Keep within realistic bounds
    
    # Generate AFTER data (with improvement)
    after_noise = np.random.normal(0, config.noise_std_after, config.n_points)
    after_spikes = generate_spikes(
        config.n_points, 
        config.spike_probability_after, 
        config.spike_magnitude_after
    )
    after = config.base_cpu_after + daily_pattern * 0.8 + after_noise + after_spikes
    after = np.clip(after, 5, 95)  # Keep within realistic bounds
    
    # Create DataFrame
    df = pd.DataFrame({
        'Time': time_labels,
        'Timestamp': time_intervals,
        'CPU_Before (%)': np.round(before, 2),
        'CPU_After (%)': np.round(after, 2),
        'Difference (%)': np.round(before - after, 2)
    })
    
    return before, after, time_labels, df


def generate_no_improvement_data(
    config: Optional[DataGenerationConfig] = None
) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    """
    Generate synthetic data where there is NO significant improvement.
    Useful for testing the statistical analysis with null hypothesis.
    
    Parameters:
    -----------
    config : DataGenerationConfig, optional
        Configuration for data generation
        
    Returns:
    --------
    Tuple containing before, after, time_labels, df
    """
    if config is None:
        config = DataGenerationConfig()
        # Make before and after nearly identical
        config.base_cpu_after = config.base_cpu_before - 2  # Tiny difference
        config.noise_std_after = config.noise_std_before
        config.spike_probability_after = config.spike_probability_before
        config.spike_magnitude_after = config.spike_magnitude_before
    
    return generate_cpu_data(config, improvement_factor=0.02)


def generate_marginal_improvement_data(
    config: Optional[DataGenerationConfig] = None
) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    """
    Generate data with statistically significant but practically small improvement.
    
    Parameters:
    -----------
    config : DataGenerationConfig, optional
        Configuration for data generation
        
    Returns:
    --------
    Tuple containing before, after, time_labels, df
    """
    if config is None:
        config = DataGenerationConfig()
        config.base_cpu_after = config.base_cpu_before - 5  # Small difference
        config.noise_std_after = config.noise_std_before * 0.95
        config.spike_probability_after = config.spike_probability_before * 0.9
    
    return generate_cpu_data(config, improvement_factor=0.1)


def generate_large_improvement_data(
    config: Optional[DataGenerationConfig] = None
) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    """
    Generate data with large, obvious improvement.
    
    Parameters:
    -----------
    config : DataGenerationConfig, optional
        Configuration for data generation
        
    Returns:
    --------
    Tuple containing before, after, time_labels, df
    """
    if config is None:
        config = DataGenerationConfig()
        config.base_cpu_after = config.base_cpu_before * 0.5  # 50% reduction
        config.noise_std_after = config.noise_std_before * 0.5
        config.spike_probability_after = config.spike_probability_before * 0.2
        config.spike_magnitude_after = config.spike_magnitude_before * 0.3
    
    return generate_cpu_data(config, improvement_factor=0.5)


def generate_variable_improvement_data(
    n_points: int = 96,
    random_seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    """
    Generate data where improvement varies by time of day.
    
    Good improvement during business hours, minimal at night.
    
    Parameters:
    -----------
    n_points : int
        Number of data points
    random_seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    Tuple containing before, after, time_labels, df
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    config = DataGenerationConfig(n_points=n_points)
    before, _, time_labels, _ = generate_cpu_data(config)
    
    # Generate after data with time-varying improvement
    after = np.zeros(n_points)
    points_per_hour = n_points / 24
    
    for i in range(n_points):
        hour = i / points_per_hour
        
        # Improvement factor varies by time of day
        if 9 <= hour < 18:  # Business hours: good improvement
            improvement = 0.35
        elif 6 <= hour < 9 or 18 <= hour < 22:  # Transition: moderate improvement
            improvement = 0.20
        else:  # Night: minimal improvement
            improvement = 0.05
        
        noise = np.random.normal(0, 5)
        after[i] = before[i] * (1 - improvement) + noise
    
    after = np.clip(after, 5, 95)
    
    df = pd.DataFrame({
        'Time': time_labels,
        'CPU_Before (%)': np.round(before, 2),
        'CPU_After (%)': np.round(after, 2),
        'Difference (%)': np.round(before - after, 2)
    })
    
    return before, after, time_labels, df


def load_from_csv(filepath: str) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    """
    Load CPU data from a CSV file.
    
    Expected columns: Time, CPU_Before (%), CPU_After (%)
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
        
    Returns:
    --------
    Tuple containing before, after, time_labels, df
    """
    df = pd.read_csv(filepath)
    
    # Handle various column naming conventions
    time_col = None
    before_col = None
    after_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if 'time' in col_lower:
            time_col = col
        elif 'before' in col_lower:
            before_col = col
        elif 'after' in col_lower:
            after_col = col
    
    if time_col is None or before_col is None or after_col is None:
        raise ValueError(
            "CSV must contain columns with 'time', 'before', and 'after' in their names"
        )
    
    time_labels = df[time_col].tolist()
    before = df[before_col].values
    after = df[after_col].values
    
    return before, after, time_labels, df


def save_to_csv(
    before: np.ndarray,
    after: np.ndarray,
    time_labels: List[str],
    filepath: str
) -> None:
    """
    Save CPU data to a CSV file.
    
    Parameters:
    -----------
    before : np.ndarray
        CPU usage before improvement
    after : np.ndarray
        CPU usage after improvement
    time_labels : List[str]
        Time labels
    filepath : str
        Output file path
    """
    df = pd.DataFrame({
        'Time': time_labels,
        'CPU_Before (%)': np.round(before, 2),
        'CPU_After (%)': np.round(after, 2),
        'Difference (%)': np.round(before - after, 2)
    })
    df.to_csv(filepath, index=False)
    print(f"Data saved to: {filepath}")


if __name__ == "__main__":
    # Example: Generate and display different scenarios
    
    print("=" * 60)
    print("SYNTHETIC DATA GENERATION EXAMPLES")
    print("=" * 60)
    
    # Standard improvement
    print("\n1. Standard Improvement Scenario")
    print("-" * 40)
    config = DataGenerationConfig(random_seed=42)
    before, after, time_labels, df = generate_cpu_data(config)
    print(f"Before: mean={np.mean(before):.2f}%, std={np.std(before):.2f}%")
    print(f"After:  mean={np.mean(after):.2f}%, std={np.std(after):.2f}%")
    print(f"Expected improvement: ~30%")
    
    # No improvement
    print("\n2. No Improvement Scenario")
    print("-" * 40)
    before_null, after_null, _, _ = generate_no_improvement_data(
        DataGenerationConfig(random_seed=42)
    )
    print(f"Before: mean={np.mean(before_null):.2f}%, std={np.std(before_null):.2f}%")
    print(f"After:  mean={np.mean(after_null):.2f}%, std={np.std(after_null):.2f}%")
    
    # Large improvement
    print("\n3. Large Improvement Scenario")
    print("-" * 40)
    before_large, after_large, _, _ = generate_large_improvement_data(
        DataGenerationConfig(random_seed=42)
    )
    print(f"Before: mean={np.mean(before_large):.2f}%, std={np.std(before_large):.2f}%")
    print(f"After:  mean={np.mean(after_large):.2f}%, std={np.std(after_large):.2f}%")
    
    # Variable improvement
    print("\n4. Variable Improvement Scenario")
    print("-" * 40)
    before_var, after_var, _, df_var = generate_variable_improvement_data(random_seed=42)
    print(f"Before: mean={np.mean(before_var):.2f}%, std={np.std(before_var):.2f}%")
    print(f"After:  mean={np.mean(after_var):.2f}%, std={np.std(after_var):.2f}%")
    
    print("\n" + "=" * 60)
    print("Sample data (first 5 rows):")
    print(df.head())
