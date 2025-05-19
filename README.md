# Deep Learning for Disease Prediction

## Overview

This project builds and evaluates deep learning models to predict medical diagnoses based on symptoms. We use a Kaggle dataset with 132 binary symptom indicators to train a multi-layer perceptron (MLP) and compare it against a logistic regression benchmark. The goal is to assist healthcare providers in making faster and more accurate initial assessments, reducing the risk of misdiagnosis.

This tool could be deployed in hospitals or online pre-visit screenings to help direct patients to the appropriate specialists.

---

## Dataset

- **Source**: [Kaggle – Disease Prediction Dataset](https://www.kaggle.com/datasets/marslinoedward/disease-prediction-data/data)
- **Training Samples**: 3,500 (after adjustment)
- **Testing Samples**: 1,500
- **Features**: 132 binary symptom columns
- **Target Variable**: `Prognosis` – 42 disease categories

Data preparation included:
- Dropping missing values
- Standardizing features (mean = 0, std = 1)
- Reshuffling and rebalancing test/train split

---

## Modeling Approach

### 1. Multi-Layer Perceptron (MLP)
Two models were built:

**Model 1:**
- Architecture: Input → 16 → Dropout(0.5) → 8 → Dropout(0.5) → Output
- Accuracy: 100%
- Loss: 0.5372 (on adjusted test set)

**Model 2:**
- Architecture: Input → 64 → Dropout(0.2) → 16 → Dropout(0.3) → 8 → Dropout(0.3) → Output
- Accuracy: 100%
- Loss: 0.2746

**Training Settings:**
- Optimizer: Adam
- Learning Rate: 0.0015
- Weight Decay: 0.001
- Epochs: 200
- Loss Function: CrossEntropyLoss

### 2. Benchmark Model: Logistic Regression
- Achieved same 100% accuracy
- Simpler and interpretable, but less flexible for future extension

---

## Results

- **Confusion Matrix** showed perfect classification across 42 categories
- **Training/Testing loss curves** confirmed convergence without overfitting
- **MLP outperforms Logistic Regression** in flexibility and future scalability

---

## Deployment Use Case

This model could support:
- **Hospital Triage**: Direct patients to correct departments based on symptom self-report
- **Online Screenings**: Improve telehealth workflows by providing preliminary suggestions

Limitations include inability to detect new/niche diseases. Continued model retraining and updated datasets are recommended for sustained performance.

---

## Files

- `Team_27_disease_predict.ipynb` – Full training and evaluation notebook
- `Project Report_Team 27.pdf` – Written report with rationale, setup, and evaluation
- `Deep Learning Project_Team27.pptx` – Slide deck summarizing architecture and deployment

---

## How to Run

1. Install requirements (PyTorch, NumPy, Pandas, etc.)
2. Open `Team_27_disease_predict.ipynb` in Jupyter Notebook or Colab
3. Run cells to train and evaluate models
4. Inspect final loss/accuracy and confusion matrix

---

## Team Members

Team 27 – Duke MQM  
- Lizzie Wang  
- Sabrina Chen  
- Demir Degirmenci  
- Parin Vora

---

## License

This project is for educational use only. Contact authors for deployment permissions.
