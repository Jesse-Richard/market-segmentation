# Market Segmentation Classification

This project demonstrates the application of machine learning models for customer segmentation using classification algorithms. The primary goal is to predict customer segments based on various features using three classification models: **Support Vector Machine (SVM)**, **Logistic Regression**, and **Decision Tree**. The project also includes techniques for handling imbalanced data and evaluating model performance with various classification metrics.

---
Find also: A presentation document in pdf format.(`MachineLearningConsultingProposal_Presentation.pdf`)
---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Requirements](#requirements)
4. [Model Implementation](#model-implementation)
5. [Model Evaluation](#model-evaluation)
6. [Results](#results)
7. [How to Run the Project](#how-to-run-the-project)
8. [License](#license)

---

## Project Overview

In this project, we utilize customer data to classify customers into different market segments. We apply three different classification algorithms: SVM, Logistic Regression, and Decision Tree. The dataset contains customer features, and the task is to predict customer segments (e.g., APAC, Africa, US, EU, etc.) based on those features.

---

## Dataset

The dataset used in this project includes the following columns:
- **Customer Features**: Various numerical and categorical features representing customer attributes.(e.g., Customer Name, Country, State, Postal Code, Region)
- **Product Features**: A variety of features representing product properties. (e.g., Order Date, Ship Date, Ship Mode, Category, Sub-Category, Order Priority)
- **Point of Sale(POS) Features**: Features representing product exchange. (e.g., OrderID, Sales, Quantity, Discount, Profit)
- **Target Column (Segment)**: The target labels representing different customer segments (e.g., APAC, Africa, EU, US).

### Data Preprocessing:
1. **Feature Encoding**: Categorical features are encoded using one-hot encoding.
2. **Scaling**: Numerical features are scaled using `StandardScaler` for better performance, especially with SVM.
3. **Handling Imbalanced Data**: We use the `class_weight='balanced'` parameter to adjust for imbalanced classes.

---

## Requirements

To run this project, the following libraries and dependencies are required:
- Python 3.7+
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `imbalanced-learn` (if using oversampling techniques)

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## Model Implementation

### 1. **Support Vector Machine (SVM)**
The SVM model is trained using the `SVC` class from `scikit-learn` with `class_weight='balanced'` to handle imbalanced data. The model's predictions are evaluated using accuracy, precision, recall, and F1-score.

### 2. **Logistic Regression**
The Logistic Regression model uses `LogisticRegression` with the `class_weight='balanced'` parameter to handle class imbalance. It is evaluated using similar metrics as the SVM model.

### 3. **Decision Tree**
The Decision Tree model is trained using the `DecisionTreeClassifier`. A hyperparameter grid search is used to find the best depth for the tree, and `class_weight='balanced'` is applied to handle class imbalance.

---

## Model Evaluation

After training the models, the following metrics are computed to evaluate performance:
- **Precision**: The ability of the model to identify relevant instances.
- **Recall**: The ability of the model to find all relevant instances.
- **F1-Score**: The harmonic mean of precision and recall.
- **Accuracy**: The percentage of correct predictions.

A confusion matrix is also generated to visualize misclassifications.

### Example Output (Classification Report):
```plaintext
Decision Tree Classification Report:
              precision    recall  f1-score   support
        APAC       1.00      1.00      1.00      3315
      Africa       0.99      1.00      1.00      1367
      Canada       1.00      1.00      1.00       103
        EMEA       1.00      0.99      0.99      1495
          EU       1.00      1.00      1.00      3018
       LATAM       1.00      1.00      1.00      3090
          US       1.00      1.00      1.00      2999
          
    accuracy                           1.00     15387
   macro avg       1.00      1.00      1.00     15387
weighted avg       1.00      1.00      1.00     15387
```

---

## Results

After applying the three models, the following insights were obtained:
- **SVM**: High accuracy and performance for the majority class. However, slightly lower recall for minority classes.
- **Logistic Regression**: Reliable and interpretable results with balanced performance across classes.
- **Decision Tree**: Best performance with minimal preprocessing, but prone to overfitting without pruning.

Each model was evaluated for its ability to handle imbalanced data using class weights and metric-based validation.

---

## How to Run the Project

1. Clone this repository:
   ```bash
   git clone https://github.com/Jesse-Richard/market-segmentation.git
   ```

2. Navigate to the project directory:
   ```bash
   cd market-segmentation
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the main script:
   ```bash
   python market_segmentation.py
   ```

This will load the dataset, preprocess the data, train the models, and display the classification results.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
