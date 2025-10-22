# ML Damage Analysis - Mehdi Dataset

Using ML on a real dataset to predict damage that is uniform-like. This project analyzes impedance data for weight prediction using machine learning models.

## Files

- `RMSD10.m` - MATLAB code by Mehdi
- `run_study.sh` - Converts `plot_explore_data.ipynb` to Python script and executes it
- `plot_explore_data.ipynb` - Main analysis notebook
- `Datafolder/` - Contains impedance sweep data (.mat files)

## Analysis Results

<img src="figures/Sample_uniform_corrosion.jpg" width="40%" alt="Sample with uniform corrosion">

<img src="figures/Scale_and_sample.jpg" width="40%" alt="Sample on the scale">


### RMSD Analysis
Root Mean Square Deviation of major peak in real part of impedance data:

<img src="figures/RMSD/rmsd_vs_weight_165_200kHz.png" width="60%" alt="RMSD vs Weight">

### ML Predictions
Models trained on real part of impedance to predict weight:

Using 150-200kHz full dataset

<img src="figures/Predictions/model_prediction_150_200kHz.png" width="60%" alt="Model Prediction 150-200kHz">

Using 50-500kHz full dataset

<img src="figures/Predictions/model_prediction_50_500kHz.png" width="60%" alt="Model Prediction 50-500kHz">

**Key Finding:** Using just the major peak vs. full dataset yields similar performance, but peak-only analysis is significantly faster.
