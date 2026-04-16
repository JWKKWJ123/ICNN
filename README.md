# Global and Local Interpretable CNN (GL-ICNN) for AD diagnosis and prediction
The core model architecture is consistent with our previous conference work, while this repository extends it with additional components for reproducibility and interpretability, including an interactive visualization module implemented in Google Colab.   
This repository extends our previous conference work: https://ieeexplore.ieee.org/abstract/document/10981153


DOI link: zenodo.org/records/19610348


## Repository Structure

- data/: saved features and example data used for testing and visualization  
- plot/: example output figures (group-level and individual-level explanations)  
- visualization/: visualization scripts (used in the Colab notebook)  
- GL_ICNN.py: model architecture  
- model_training.py: training pipeline  
- model_testing.py: testing and evaluation pipeline  
- requirements.txt: required Python packages  


## Model Building

The model is implemented using the PyTorch framework.  

GL_ICNN.py defines the GL-ICNN model, including:
- multi-branch CNN feature extractors (global and local pathways)  
- an interpretable output block based on Explainable Boosting Machine (EBM)  


## Model Training

The GL-ICNN is trained in two stages:

1. CNN pretraining  
2. Alternating optimization of CNN and EBM components  

model_training.py:
- trains the model  
- saves the best model  
- outputs learning curves and extracted features  


## Model Testing

Due to file size limitations, the trained model (.pth) is not included in this repository.  
Instead, we provide the extracted CNN features (for both training and testing sets) in the data/ folder.

model_testing.py:
- trains the EBM based on extracted CNN features  
- evaluates performance on internal/external test sets  
- computes:
  - performance metrics  
  - group-level feature importance  
  - individual-level feature contributions  

Generated figures are saved in the plot/ folder (example results are already provided).



## Code Availability

A stable archived version of this repository is available on Zenodo:  
https://zenodo.org/records/19610348  

This archived version corresponds to the code used in the journal study and ensures reproducibility of the reported results.


## Visualization (Colab)

The interactive visualization of the heatmaps is available via Google Colab:

https://colab.research.google.com/drive/1ULK1Utmp90f9NvWgjuCdKG_jzQzStYvJ?usp=sharing  

NOTE: Please scroll down to access the interactive visualizations. Any initial warnings can be safely ignored.

### Description

The visualization module enables interactive exploration of 3D heatmaps across anatomical planes (axial, coronal, sagittal).

- Implemented in visualization/visualization.py  
- Designed for use within the Colab notebook (not as a standalone script)  

It supports:
- slice-wise navigation across 3D volumes  
- overlay of heatmaps on anatomical images  
- comparison across different models  

For best user experience, please use the Colab notebook rather than running locally.


## Notes

- This repository focuses on reproducibility of the model and interpretability pipeline.  
- The Colab notebook is recommended for interactive visualization.  
- Large model files are not included due to storage limitations.  















