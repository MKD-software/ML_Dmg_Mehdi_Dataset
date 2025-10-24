"""
Usage example for data_prep.py functions.
"""

from data_prep import prepare_training_data, prepare_evaluation_data
import pandas as pd

# Example usage:

# 1. Prepare training data
# training_result = prepare_training_data(
#     datasets=datasets,  # Your loaded datasets from .mat files
#     freq_start_khz=165,
#     freq_stop_khz=200
# )
# 
# X_train = training_result['X_data']
# y_train = training_result['y_data']
# scaler = training_result['scaler']
# reference_data = training_result['reference_data']

# 2. Prepare evaluation data
# start_date = pd.Timestamp('2025-10-15')
# end_date = pd.Timestamp('2025-10-22 23:59:59')
# 
# evaluation_result = prepare_evaluation_data(
#     real_impedance=real_impedance,      # Your loaded real impedance DataFrame
#     imag_impedance=imag_impedance,      # Your loaded imaginary impedance DataFrame  
#     frequencies=frequencies,            # Your frequency array
#     sample_name="sample_6",
#     scaler=scaler,                      # From training
#     reference_data=reference_data,      # From training
#     freq_start_khz=165,
#     freq_stop_khz=200,
#     freq_offset=20,
#     date_range=(start_date, end_date)
# )
# 
# X_eval = evaluation_result['X_data']
# timestamps = evaluation_result['timestamps']