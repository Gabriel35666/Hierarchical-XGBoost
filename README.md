# Hierarchical XGBoost Lithology Classification Method

This repository contains the source code and datasets used in the study:

**Hierarchical XGBoost Lithology Classification Method Incorporating Geological Constraints**

The proposed method introduces geological prior knowledge into the lithology classification process by constructing a hierarchical framework based on lithology families. The approach first distinguishes between clastic and carbonate rock families and then performs fine-grained lithology classification using dedicated sub-models.

The method improves classification accuracy, enhances geological consistency, and demonstrates stronger cross-well generalization ability compared with conventional machine learning approaches.

# Dataset Description

The `dataset` folder contains well log datasets used in this study.

|    File   |                          Description                    |
|-----------|---------------------------------------------------------|
| train.csv | Training dataset used to train the classification model |
| test1.csv |                 Test dataset from Well 1                |
| test2.csv |                 Test dataset from Well 2                |
| test3.csv |                 Test dataset from Well 3                |

## 1. Cross-well testing

Run the following scripts to evaluate the model on different test wells:
main_test1.py, main_test2.py, main_test3.py

These scripts perform independent cross-well validation experiments.

## 2. Mainstream model comparison

This script compares the proposed hierarchical XGBoost method with several mainstream machine learning models.

---

## 3. Geological constraint validation experiment

This experiment verifies the effectiveness of introducing geological constraints into the lithology classification model.

---

## 4. Threshold prior experiment


This experiment investigates the influence of threshold-based geological prior information on model performance.

---

# Method Workflow

The workflow of the proposed method includes the following steps:

1. Data preprocessing and feature normalization  
2. Lithology family classification (clastic vs carbonate)  
3. Construction of specialized sub-models for different lithology families  
4. Fine-grained lithology classification  
5. Cross-well validation and performance evaluation

---

# Evaluation Metrics

The model performance is evaluated using the following metrics:

- Macro F1 Score  
- Weighted F1 Score  
- Cohen's Kappa Coefficient  
- Classification Accuracy

---

# Reproducibility

All experiments reported in the paper can be reproduced using the scripts provided in this repository. Random seeds are fixed to ensure consistent results.

---






