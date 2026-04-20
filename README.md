# Unsupervised LSTM Autoencoders for Interpretable Temporal Structures in Financial Time-Series Data using XAI Techniques

## Requirements
 
Python 3.11, CPU only.
 
```bash
pip install torch numpy scikit-learn scipy pandas matplotlib seaborn statsmodels hmmlearn

```

## Running the pipeline
 
To reproduce the paper's results in full, the following scripts should be executed in order. Each stage writes intermediate artefacts to structured output directories, allowing subsequent stages to be re-run
independently.

 
```bash
# Preprocessing and model training
python preprocess.py
python train.py
 
# Track 1 - cluster membership
python clustering_analysis.py
python vector_shap_track1.py
python timeshap_track1.py
 
# Track 2 - reconstruction error
python error_segmentation_analysis.py
python vector_shap_track2.py
python timeshap_track2.py
 
# Evaluation, baselines, figures
python faithfulness.py
python regime_shap_analysis.py
python vix_validation.py
python benchmark_baselines.py
python visualise_vector_shap_track1.py
python visualise_vector_shap_track2.py
python visualise_timeshap_track1.py
python visualise_timeshap_track2.py
```
 