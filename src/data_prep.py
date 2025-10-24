"""
Minimal data preparation functions for impedance analysis datasets.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def prepare_dataset(data_source, datasets=None, real_impedance=None, imag_impedance=None, 
                   frequencies=None, freq_start_khz=165, freq_stop_khz=200, 
                   freq_offset=0, scaler=None, reference_data=None, 
                   sample_name=None, date_range=None):
    """
    Minimal data preparation function for both training and evaluation datasets.
    
    Parameters:
    -----------
    data_source : str
        "mat_files" or "accelerated_corrosion"
    datasets : list
        List of dataset dictionaries (for mat_files)
    real_impedance : pd.DataFrame  
        Real impedance data (for accelerated_corrosion)
    imag_impedance : pd.DataFrame
        Imaginary impedance data (for accelerated_corrosion)
    frequencies : np.array
        Frequency array (for accelerated_corrosion)
    freq_start_khz, freq_stop_khz : float
        Frequency range in kHz
    freq_offset : float
        Frequency offset in kHz
    scaler : StandardScaler or None
        Pre-fitted scaler or None to create new
    reference_data : dict or None
        Reference data for normalization
    sample_name : str
        Sample name for accelerated corrosion data
    date_range : tuple
        (start_date, end_date) for filtering
    
    Returns:
    --------
    dict: X_data, y_data, scaler, reference_data, timestamps
    """
    
    X_data, y_data, timestamps_list = [], [], []
    
    # Adjust frequency range
    freq_start = freq_start_khz + freq_offset
    freq_stop = freq_stop_khz + freq_offset
    
    if data_source == "mat_files":
        # Training data from .mat files
        if datasets is None:
            raise ValueError("datasets must be provided for mat_files")
            
        # Get reference from first dataset
        if reference_data is None:
            freq_khz_ref = datasets[0]["f"] / 1000
            freq_mask_ref = (freq_khz_ref >= freq_start) & (freq_khz_ref <= freq_stop)
            reference_data = {
                'R_ref': datasets[0]["R"][freq_mask_ref],
                'Xc_ref': datasets[0]["Xc"][freq_mask_ref]
            }
        
        # Process datasets
        freqs = datasets[0]["f"]
        for d in datasets:
            freq_khz = d["f"] / 1000
            freq_mask = (freq_khz >= freq_start) & (freq_khz <= freq_stop)
            
            R = d["R"][freq_mask]
            Xc = d["Xc"][freq_mask]
            freq_feat = freqs[freq_mask]
            
            features = np.concatenate([R, Xc, freq_feat])
            X_data.append(features)
            y_data.append(d["weight_g"] - 115.0)
            
    elif data_source == "accelerated_corrosion":
        # Evaluation data from accelerated corrosion test
            
        # Get sample data
        sample_data = real_impedance[sample_name].dropna()
        imag_sample_data = imag_impedance[sample_name].dropna()
        
        # Apply date filtering
        if date_range is not None:
            start_date, end_date = date_range
            date_mask = (sample_data.index >= start_date) & (sample_data.index <= end_date)
            sample_data = sample_data[date_mask]
            imag_sample_data = imag_sample_data[date_mask]
        
        # Frequency filtering
        freq_khz = frequencies / 1000
        freq_mask = (freq_khz >= freq_start) & (freq_khz <= freq_stop)
        
        # Get reference if not provided
        if reference_data is None:
            first_R = sample_data.iloc[0][freq_mask]
            first_Xc = imag_sample_data.iloc[0][freq_mask]
            reference_data = {'R_ref': first_R, 'Xc_ref': first_Xc}
        
        # Process each timestamp
        for timestamp, impedance_array in sample_data.items():
            R = impedance_array[freq_mask]
            Xc = imag_sample_data.loc[timestamp][freq_mask]
            freq_feat = frequencies[freq_mask]
            
            features = np.concatenate([R, Xc, freq_feat])
            X_data.append(features)
            timestamps_list.append(timestamp)
            
        y_data = None
    
    else:
        raise ValueError(f"Unknown data_source: {data_source}")
    
    # Convert to arrays and apply scaling
    X_data = np.array(X_data, dtype=np.float32)
    if y_data is not None:
        y_data = np.array(y_data, dtype=np.float32).reshape(-1, 1)
    
    if scaler is None:
        scaler = StandardScaler()
        X_data = scaler.fit_transform(X_data)
    else:
        X_data = scaler.transform(X_data)
    
    return {
        'X_data': X_data,
        'y_data': y_data,
        'scaler': scaler,
        'reference_data': reference_data,
        'timestamps': timestamps_list if timestamps_list else None
    }


def prepare_training_data(datasets, freq_start_khz=165, freq_stop_khz=200):
    """Simplified function for training data preparation."""
    return prepare_dataset(
        data_source="mat_files",
        datasets=datasets,
        freq_start_khz=freq_start_khz,
        freq_stop_khz=freq_stop_khz
    )


def prepare_evaluation_data(real_impedance, imag_impedance, frequencies, 
                          sample_name, scaler, reference_data,
                          freq_start_khz=165, freq_stop_khz=200, 
                          freq_offset=20, date_range=None):
    """Simplified function for evaluation data preparation."""
    return prepare_dataset(
        data_source="accelerated_corrosion",
        real_impedance=real_impedance,
        imag_impedance=imag_impedance,
        frequencies=frequencies,
        freq_start_khz=freq_start_khz,
        freq_stop_khz=freq_stop_khz,
        freq_offset=freq_offset,
        scaler=scaler,
        reference_data=reference_data,
        sample_name=sample_name,
        date_range=date_range
    )