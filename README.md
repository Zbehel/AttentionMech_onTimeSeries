# 🧠 AttentionMech_onTimeSeries

## 📝 Description

This project is a binary classification among Epileptic patients and non epileptic. It is trained on EEG recordings from the TUH Dataset.
The whole project can be find here : [Original Repository](https://github.com/sonydata/EEG_Epilepsy_Classification)

And with a streamlit deployed on HF : [EEG Epilepsy App](https://huggingface.co/spaces/MorganBrizon/EEG_Epilepsy_App)

## 🤖 Model

Here is a reimplementation of EpilepsyNet a model based on Attention among the correlation between each channel of acquisition.
The original article from Oh Shu Lih et al. here -> [Link](https://www.sciencedirect.com/science/article/pii/S0010482523007771?ref=pdf_download&fr=RR-2&rr=92b0ae604bfc9a80)

## ⚙️ Method

The method used in this project involves the following steps:

1.  **Data Preprocessing:** The EEG recordings from the TUH dataset are preprocessed to remove noise and artifacts. A selection of n=5 acquitision per patient (adjustable) is made to preserve balanced dataset. 

1. 🔄 Preprocessing Pipeline

Raw .edf EEG signals were cleaned and segmented using the main following steps:

        1.    Load session from .edf (via MNE)
        2.    Apply bandpass filter (1–45 Hz)
        3.    Resample to 250 Hz
        4.    Select relevant EEG channels
        5.    Select a 60 seconds windows
        6.    Fragment the 1 minute into 12 splits of 5 seconds
        6.    Normalize each epoch

2.  **Feature Extraction:** Matrix Correlations between relevant channels is computed, and the upper triangle is flattened becoming the inputs for the model.

3.  **Model Training:** The EpilepsyNet model is trained using positional embedding, attention mechanism and two Dense layers before classification corresponding to labels (epileptic or non-epileptic).
4.  **Model Evaluation:** The trained model is evaluated on a held-out test set to assess its performance.



## ✅ Results

| Model        | Sensitivity | Specificity|
|--------------|-------------|----------|
| EpilepsyNet  | 82%         | 71%     |

with $Sensitivity = \frac {TP} {TP+FN}$ & $Specificity = \frac {FP}{FP+TN}$

The EpilepsyNet model achieves high accuracy in classifying epileptic and non-epileptic EEG recordings. The results demonstrate the effectiveness of the attention mechanism in capturing the correlations between EEG channels.
Even with a Dataset of that rare quality and size for the specific task, the model seems to overfit, more recordings from diverse subjects would delay the model's end of learning.