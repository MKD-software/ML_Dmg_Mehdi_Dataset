"""
Accelerated Corrosion Test Data Loader

This module provides functions to load and process accelerated corrosion test data
from CSV files organized in timestamped directories.

Data structure: Accelerated corrosion test data/YYYY-MM-DD_HH-MM-SS/sample_X.csv
- CSV columns: Frequency (Hz), Impedance (ohms), Phase (Radians), Temperature (C), Humidity (%)
- Note: "Impedance (ohms)" is actually the Real part, "Phase (Radians)" is the Imaginary part
- Temperature and humidity columns are ignored
- Data is filtered to 50kHz-500kHz range and interpolated to 9000 evenly spaced points
"""

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
from scipy import interpolate
import warnings

def load_accelerated_corrosion_data(data_dir="Accelerated corrosion test data", 
                                  freq_min=50000, freq_max=500000, num_points=9000):
    """
    Load accelerated corrosion test data from CSV files.
    
    Parameters:
    -----------
    data_dir : str
        Path to the directory containing timestamped subdirectories
    freq_min : float
        Minimum frequency in Hz (default: 50kHz)
    freq_max : float
        Maximum frequency in Hz (default: 500kHz)  
    num_points : int
        Number of evenly spaced frequency points (default: 9000)
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'frequency': numpy array of frequency points
        - 'real_data': DataFrame with real impedance data (samples as columns, timestamps as index)
        - 'imag_data': DataFrame with imaginary impedance data (samples as columns, timestamps as index)
        - 'timestamps': pandas DatetimeIndex of measurement timestamps
        - 'sample_names': list of available sample names
    """
    
    # Get absolute path to data directory
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(os.path.dirname(__file__), '..', data_dir)
    data_dir = os.path.abspath(data_dir)
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Create target frequency array
    freq_target = np.linspace(freq_min, freq_max, num_points)
    
    # Get all timestamped subdirectories
    subdirs = [d for d in os.listdir(data_dir) 
               if os.path.isdir(os.path.join(data_dir, d)) and 
               d.replace('-', '').replace('_', '').replace(':', '').isdigit()]
    
    if not subdirs:
        raise ValueError(f"No timestamped subdirectories found in {data_dir}")
    
    subdirs.sort()  # Sort chronologically
    
    # Parse timestamps from directory names
    timestamps = []
    for subdir in subdirs:
        try:
            # Parse format: YYYY-MM-DD_HH-MM-SS
            timestamp = datetime.strptime(subdir, '%Y-%m-%d_%H-%M-%S')
            timestamps.append(timestamp)
        except ValueError:
            warnings.warn(f"Could not parse timestamp from directory: {subdir}")
            continue
    
    # Find all unique sample files across all directories
    all_samples = set()
    for subdir in subdirs:
        subdir_path = os.path.join(data_dir, subdir)
        csv_files = glob.glob(os.path.join(subdir_path, "sample_*.csv"))
        for csv_file in csv_files:
            sample_name = os.path.basename(csv_file).replace('.csv', '')
            all_samples.add(sample_name)
    
    sample_names = sorted(list(all_samples))
    
    if not sample_names:
        raise ValueError("No sample CSV files found in any subdirectory")
    
    print(f"Found {len(sample_names)} unique samples: {sample_names}")
    print(f"Found {len(subdirs)} timestamped directories")
    
    # Initialize data storage
    real_data = pd.DataFrame(index=pd.DatetimeIndex(timestamps), columns=sample_names)
    imag_data = pd.DataFrame(index=pd.DatetimeIndex(timestamps), columns=sample_names)
    
    # Process each timestamp directory
    for i, (subdir, timestamp) in enumerate(zip(subdirs, timestamps)):
        subdir_path = os.path.join(data_dir, subdir)
        print(f"Processing {subdir} ({i+1}/{len(subdirs)})")
        
        # Process each sample file
        for sample_name in sample_names:
            csv_file = os.path.join(subdir_path, f"{sample_name}.csv")
            
            if not os.path.exists(csv_file):
                # File missing, leave as NaN
                continue
                
            try:
                # Load CSV data
                df = pd.read_csv(csv_file)
                
                # Extract frequency, real (impedance), and imaginary (phase) parts
                freq = df['Frequency (Hz)'].values
                real_part = df['Impedance (ohms)'].values  # Actually real part
                imag_part = df['Phase (Radians)'].values   # Actually imaginary part
                
                # Filter to frequency range
                freq_mask = (freq >= freq_min) & (freq <= freq_max)
                freq_filtered = freq[freq_mask]
                real_filtered = real_part[freq_mask]
                imag_filtered = imag_part[freq_mask]
                
                if len(freq_filtered) < 2:
                    warnings.warn(f"Insufficient data points in frequency range for {csv_file}")
                    continue
                
                # Interpolate to target frequency grid
                # Remove any duplicate frequencies for interpolation
                unique_indices = np.unique(freq_filtered, return_index=True)[1]
                freq_unique = freq_filtered[unique_indices]
                real_unique = real_filtered[unique_indices]
                imag_unique = imag_filtered[unique_indices]
                
                # Sort by frequency for interpolation
                sort_indices = np.argsort(freq_unique)
                freq_sorted = freq_unique[sort_indices]
                real_sorted = real_unique[sort_indices]
                imag_sorted = imag_unique[sort_indices]
                
                # Interpolate real and imaginary parts
                f_real = interpolate.interp1d(freq_sorted, real_sorted, 
                                            kind='linear', bounds_error=False, fill_value=np.nan)
                f_imag = interpolate.interp1d(freq_sorted, imag_sorted, 
                                            kind='linear', bounds_error=False, fill_value=np.nan)
                
                real_interp = f_real(freq_target)
                imag_interp = f_imag(freq_target)
                
                # Store interpolated data
                real_data.loc[timestamp, sample_name] = real_interp
                imag_data.loc[timestamp, sample_name] = imag_interp
                
            except Exception as e:
                warnings.warn(f"Error processing {csv_file}: {str(e)}")
                continue
    
    # Remove completely empty samples
    real_data = real_data.dropna(axis=1, how='all')
    imag_data = imag_data.dropna(axis=1, how='all')
    
    # Ensure both dataframes have the same columns
    common_samples = list(set(real_data.columns) & set(imag_data.columns))
    real_data = real_data[common_samples]
    imag_data = imag_data[common_samples]
    
    print(f"Successfully loaded data for {len(common_samples)} samples")
    print(f"Data shape: {real_data.shape} (timestamps x samples)")
    print(f"Frequency range: {freq_min/1000:.0f} - {freq_max/1000:.0f} kHz with {num_points} points")
    
    return {
        'frequency': freq_target,
        'real_data': real_data,
        'imag_data': imag_data,
        'timestamps': real_data.index,
        'sample_names': common_samples
    }


def get_complex_impedance(data):
    """
    Combine real and imaginary data into complex impedance.
    
    Parameters:
    -----------
    data : dict
        Output from load_accelerated_corrosion_data()
        
    Returns:
    --------
    pandas.DataFrame
        Complex impedance data with the same structure as real_data/imag_data
    """
    return data['real_data'] + 1j * data['imag_data']


def get_magnitude_phase(data):
    """
    Convert real/imaginary data to magnitude and phase.
    
    Parameters:
    -----------
    data : dict
        Output from load_accelerated_corrosion_data()
        
    Returns:
    --------
    tuple
        (magnitude_data, phase_data) as pandas DataFrames
    """
    complex_impedance = get_complex_impedance(data)
    magnitude = np.abs(complex_impedance)
    phase = np.angle(complex_impedance)
    
    return magnitude, phase


# Example usage and test function
def test_data_loading():
    """Test function to verify data loading works correctly."""
    try:
        data = load_accelerated_corrosion_data()
        
        print("\n=== Data Loading Test Results ===")
        print(f"Frequency array shape: {data['frequency'].shape}")
        print(f"Real data shape: {data['real_data'].shape}")
        print(f"Imaginary data shape: {data['imag_data'].shape}")
        print(f"Number of timestamps: {len(data['timestamps'])}")
        print(f"Number of samples: {len(data['sample_names'])}")
        print(f"Sample names: {data['sample_names']}")
        print(f"Timestamp range: {data['timestamps'].min()} to {data['timestamps'].max()}")
        
        # Check for any data
        real_not_null = data['real_data'].notna().sum().sum()
        imag_not_null = data['imag_data'].notna().sum().sum()
        print(f"Non-null real data points: {real_not_null}")
        print(f"Non-null imaginary data points: {imag_not_null}")
        
        return True
        
    except Exception as e:
        print(f"Data loading test failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Run test when script is executed directly
    test_data_loading()
