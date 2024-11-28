# Black_Friday_Sales_Analysis

#### Brief History: 
Black Friday, traditionally observed the day after Thanksgiving in the United States, marks the unofficial start of the holiday shopping season .Black Friday has evolved into a major retail event, both in physical stores and online. Retailers later rebranded the term to reflect the point at which they moved from being in the red (loss) to being in the black (profit) due to increased sales.

## **Overview**

This repository contains a machine learning model built in Python using the train.csv dataset. It involves data exploration with pandas, including cleaning, handling missing values, and feature engineering. The model is trained with scikit-learn, featuring dataset splitting, algorithm selection, and fitting. Performance is evaluated using accuracy and confusion matrix metrics.

Dataset: train.csv used for training the model.

### **Key Steps:**

Data exploration and cleaning
Feature engineering
Model training and evaluation
Analysis Stages

---

---> **Data Loading and Inspection**:

Load the dataset using pandas.
Inspect the structure and summary statistics.

```python

import pandas as pd

# Load the dataset
data = pd.read_csv('train.csv')

# Display the first few rows and summary statistics
print(data.head())
print(data.describe())
```

---> **Model Training** :

Split the dataset into training and testing sets.
Train a model using scikit-learn.
Evaluate performance with accuracy and confusion matrix.


```python


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Split the data
X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
```


### **Contact** :
[ Linkedin ](https://www.linkedin.com/in/rachuri-vijaykumar/) // [ Github ](https://github.com/vijaykumarr1452) // [ Email ](mailto:vijaykumarit45@gmail.com) // [ Twitter ](https://x.com/vijay_viju1)
