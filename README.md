# ML Projects Repository

This repository contains two machine learning projects: Cardiovascular Disease (CVD) prediction and Housing price analysis.

---

## 📁 Project Structure

```
├── CVD_Project_Code.py          # CVD classification model
├── CVD_cleaned.csv              # CVD dataset
├── Housing.ipynb                # Housing analysis notebook
├── Housing.csv                  # Housing dataset
├── README.md                    # This file
└── .gitignore                   # Git ignore rules
```

---

## 🔴 Project 1: Cardiovascular Disease (CVD) Prediction

### Overview
This project builds and compares multiple machine learning models to predict cardiovascular disease based on patient health data.

### Dataset
- **File**: `CVD_cleaned.csv`
- **Target Variable**: `General_Health`
- Cleaned and preprocessed CVD health data

### Models Implemented
1. **Support Vector Machine (SVM)**
2. **Random Forest Classifier**
3. **Logistic Regression**

### Features
- Data preprocessing with categorical encoding (OneHotEncoder)
- Train-test split (80:20) with stratification
- Model evaluation with accuracy score, classification report, and confusion matrix
- Model comparison visualization
- Hyperparameter tuning using RandomizedSearchCV for Random Forest
- Performance metrics and accuracy comparison bar chart

### Usage
```bash
python CVD_Project_Code.py
```

### Dependencies
- pandas
- numpy
- scikit-learn
- matplotlib

---

## 🏠 Project 2: Housing Price Analysis

### Overview
This project analyzes housing data and creates visualizations based on price ranges and property features.

### Dataset
- **File**: `Housing.csv`
- Contains housing data with various attributes and prices

### Analysis Included
1. **Price Range Categorization**: Classifies houses into price ranges (in Indian Lakhs)
   - 0-25 Lakhs
   - 26-50 Lakhs
   - 51-75 Lakhs
   - 76-100 Lakhs
   - >100 Lakhs

2. **Visualizations**
   - Line chart showing distribution across price ranges
   - AC vs. Non-AC average price comparison
   - Additional exploratory data analysis

### Usage
Open the Jupyter notebook:
```bash
jupyter notebook Housing.ipynb
```

### Dependencies
- pandas
- matplotlib
- seaborn
- jupyter

---

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- pip (Python package manager)
- Jupyter Notebook (for Housing project)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ml-projects.git
cd ml-projects
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

---

## 📊 Results & Model Performance

### CVD Project
The script trains three models and compares their accuracy:
- Outputs accuracy scores, classification reports, and confusion matrices
- Generates a bar chart comparing model accuracies
- Performs hyperparameter tuning for Random Forest

### Housing Project
Generates visualizations and statistical analysis:
- Distribution of houses across price ranges
- Average prices for different property features
- Data-driven insights for housing market analysis

---

## 📝 Notes

- All datasets are included in the repository
- Data preprocessing is handled within the scripts
- Models are evaluated using appropriate metrics
- Results include confusion matrices and classification reports

---

## 👤 Author
om ghate

---

## 📄 License
This project is open source and available under the MIT License.

---

## 💡 Future Improvements
- Add cross-validation for more robust model evaluation
- Implement additional algorithms (Gradient Boosting, Neural Networks)
- Add feature importance analysis
- Create API endpoints for model predictions
- Add unit tests and documentation
- Implement data visualization dashboard

---

## 📧 Questions or Suggestions?
Feel free to open an issue or contact the repository maintainer.
