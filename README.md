# credit-card-fraud-detection-using-deep-learning
A deep learning project that detects fraudulent credit card transactions using an ANN model and threshold analysis on the Kaggle fraud dataset.

## Project Overview

This project applies deep learning to detect fraudulent credit card transactions. It is a binary classification problem where:

- **0 = normal transaction**
- **1 = fraudulent transaction**

I chose this topic because fraud detection is a practical and important problem in finance. Even though fraudulent transactions make up only a very small part of the dataset, missing them can lead to significant financial loss. This makes the task both challenging and meaningful.

---

## Dataset

The project uses the **Credit Card Fraud Detection** dataset from Kaggle.  
The dataset includes:

- `Time`
- `Amount`
- `V1` to `V28`
- `Class` as the target variable

A key challenge of this dataset is that it is **highly imbalanced**, with fraudulent transactions representing only a very small percentage of all records.

### Class Distribution

![Class Distribution](images/class_distribution.png)

This chart clearly shows that fraud cases are much fewer than non-fraud cases. Because of this imbalance, accuracy alone is not enough to judge model performance.

---

## Data Exploration

### Transaction Amount Distribution

![Transaction Amount Distribution](images/amount_distribution.png)

Most transactions have relatively small values, while only a small number have large amounts.

### Transaction Time Distribution

![Transaction Time Distribution](images/time_distribution.png)

This figure shows how transactions are distributed over time.

### Correlation Heatmap

![Correlation Heatmap](images/correlation_heatmap.png)

The heatmap provides an overview of the relationships between variables in the dataset.

---

## Preprocessing

The main preprocessing steps were:

- checking dataset shape
- checking missing values
- removing duplicate rows
- splitting features and target
- scaling `Time` and `Amount`
- splitting the data into training and testing sets

I only scaled `Time` and `Amount` because the other features had already been transformed in the original dataset.

---

## Model

I used **(ANN)** model for this project.

This model was chosen because it can learn non-linear patterns in numerical data and works well for binary classification tasks.

---

## Training Results

### Training and Validation Loss

![Training and Validation Loss](images/training_loss.png)

The loss curves decrease over time, which indicates that the model learned useful patterns from the data and improved during training.

### Training and Validation Accuracy

![Training and Validation Accuracy](images/training_accuracy.png)

The accuracy curves show that the model achieved strong overall performance. However, because the dataset is highly imbalanced, accuracy is not the most important metric.

---

## Threshold Analysis

To better evaluate the model, I compared predictions at different thresholds.

### Threshold = 0.5

![Confusion Matrix 0.5](images/confusion_matrix_05.png)

This is the default threshold and provides a basic balance between fraud detection and false alarms.

### Threshold = 0.3

![Confusion Matrix 0.3](images/confusion_matrix_03.png)

A lower threshold makes the model more sensitive to fraud. This usually improves recall, meaning more fraud cases are detected, but it can also increase false positives.

### Threshold = 0.8

![Confusion Matrix 0.8](images/confusion_matrix_08.png)

A higher threshold makes the model more conservative. This may reduce false positives, but it can also miss more fraud cases.

---

## Key Insight

The most important lesson from this project is that **accuracy alone is not enough** for fraud detection.

Because the dataset is extremely imbalanced:

- a **lower threshold** can help detect more fraudulent transactions
- a **higher threshold** can help reduce false alarms

This means threshold selection should depend on the real business objective.

---

## Conclusion

This project shows that deep learning can be applied effectively to credit card fraud detection. The ANN model was able to learn patterns from the data and produce useful results, especially when combined with threshold analysis.

Overall, this project highlights the importance of choosing suitable evaluation metrics and understanding the trade-off between detecting more fraud cases and reducing false alerts.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow / Keras
