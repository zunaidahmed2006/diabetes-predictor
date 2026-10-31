# 🩺 Diabetes Prediction System using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org/)

An end-to-end Machine Learning classification project designed to predict whether a patient has diabetes based on diagnostic medical measurements. The model is built using a **Support Vector Machine (SVM)** classifier with feature standardization.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Workflow & Methodology](#-workflow--methodology)
- [Model Performance](#-model-performance)
- [Predictive System](#-predictive-system)
- [Repository Structure](#-repository-structure)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Future Enhancements](#-future-enhancements)

---

## 📖 Project Overview

Diabetes is a chronic metabolic disease characterized by elevated levels of blood glucose. Early detection can prevent severe complications such as heart disease, kidney damage, and nerve damage.

This project implements a machine learning pipeline using Python to predict diabetes onset using diagnostic metrics. The pipeline covers:
- Data inspection and statistical analysis
- Feature scaling via `StandardScaler`
- Stratified train-test dataset split
- Model training with a Linear Support Vector Classifier (`SVC`)
- Real-time custom input inference

---

## 📊 Dataset

The project utilizes the **Pima Indians Diabetes Dataset** originally provided by the National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK).

### Feature Details:

| Feature | Description |
| :--- | :--- |
| **`Pregnancies`** | Number of times pregnant |
| **`Glucose`** | Plasma glucose concentration over 2 hours in an oral glucose tolerance test |
| **`BloodPressure`** | Diastolic blood pressure ($mm\ Hg$) |
| **`SkinThickness`** | Triceps skin fold thickness ($mm$) |
| **`Insulin`** | 2-Hour serum insulin ($mu\ U/ml$) |
| **`BMI`** | Body Mass Index ($weight\ in\ kg / (height\ in\ m)^2$) |
| **`DiabetesPedigreeFunction`**| Genetic score indicating history of diabetes in family |
| **`Age`** | Age in years |
| **`Outcome`** *(Target)* | `0` = Non-Diabetic, `1` = Diabetic |

### Class Distribution:
- **Total Samples:** 768
- **Non-Diabetic (`0`):** 500 instances (~65.1%)
- **Diabetic (`1`):** 268 instances (~34.9%)

---

## ⚙️ Workflow & Methodology

```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│  Raw Data       │ ──> │ Data Preprocessing   │ ──> │ Feature Scaling     │
│  (diabetes.csv) │     │ & Inspection         │     │ (StandardScaler)    │
└─────────────────┘     └──────────────────────┘     └─────────────────────┘
                                                                │
                                                                ▼
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│ Prediction /    │ <── │ Model Evaluation     │ <── │ Model Training      │
│ Inference       │     │ (Accuracy Score)     │     │ (SVM Linear Kernel) │
└─────────────────┘     └──────────────────────┘     └─────────────────────┘
```

1. **Data Preprocessing**:
   - Cleaned headers and validated column types.
   - Evaluated feature means grouped by outcome to identify health trends (e.g., higher mean Glucose and BMI observed in diabetic cases).
2. **Feature Scaling**:
   - Standardized features with `StandardScaler` so that each feature has zero mean and unit variance, preventing features with large scales from dominating the SVM optimization.
3. **Train-Test Split**:
   - 80% training data (614 samples) and 20% test data (154 samples).
   - Preserved class proportions across both splits using `stratify=y`.
4. **Model Architecture**:
   - Trained a **Support Vector Machine (SVM)** classifier using a linear kernel (`svm.SVC(kernel='linear')`).

---

## 📈 Model Performance

| Dataset Split | Accuracy Score |
| :--- | :--- |
| **Training Set** | **78.66%** |
| **Testing Set** | **77.27%** |

The consistent performance across both training and test sets indicates good generalization without severe overfitting.

---

## 🔮 Predictive System

The model allows inference on new patient data. Example prediction workflow:

```python
import numpy as np

# Sample health metric: (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age)
input_data = (4, 110, 92, 0, 0, 37.6, 0.191, 30)

# Reshape and standardize input
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
std_data = scalar.transform(input_data_reshaped)

# Predict
prediction = classifier.predict(std_data)

if prediction[0] == 0:
    print("The person is not diabetic.")
else:
    print("The person is diabetic.")
```

---

## 📁 Repository Structure

```text
ml-diabetes-predictor/
├── ml project 1.ipynb      # Main Jupyter notebook with data analysis, modeling & prediction
├── README.md               # Project documentation and guide
└── diabetes.csv            # Dataset file (optional / add your local copy)
```

---

## 🚀 Installation & Setup

### 1. Clone or Download the Repository
```bash
git clone https://github.com/<your-username>/ml-diabetes-predictor.git
cd ml-diabetes-predictor
```

### 2. Create and Activate a Virtual Environment (Optional but recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install numpy pandas scikit-learn jupyter
```

---

## 💻 Usage

1. Open the Jupyter Notebook environment:
   ```bash
   jupyter notebook
   ```
2. Open [`ml project 1.ipynb`](ml%20project%201.ipynb).
3. If running locally, ensure `diabetes.csv` is in the same directory or update the path in:
   ```python
   diabetes_data = pd.read_csv('diabetes.csv')
   ```
4. Run all cells sequentially to view data analysis, model training, evaluation metrics, and the prediction demo.

---

## 🌟 Future Enhancements

- [ ] Handle biological zero values (impute missing values for Glucose, Insulin, Skin Thickness, etc.).
- [ ] Experiment with other algorithms (Random Forest, XGBoost, Logistic Regression, KNN).
- [ ] Hyperparameter tuning using `GridSearchCV`.
- [ ] Build an interactive web interface using **Streamlit** or **Gradio** for user-friendly medical predictions.
- [ ] Deploy the application as a REST API using **FastAPI** or **Flask**.

---

## 📄 License

This project is licensed under the MIT License - feel free to use and modify for educational and personal projects.

