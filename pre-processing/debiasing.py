import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import xgboost as xgb  # Import XGBoost
from utils import generate_data, Reweightning, Messaging

data = generate_data()

# Split into features and labels
X = data[['age', 'credit_amount', 'sex']]
y = data['label']

# Create biased labels based on the 'sex' feature
# Let's say males (sex=1) are more likely to have a label of 1, and females (sex=0) are more likely to have a label of 0
data['label'] = np.where(data['sex'] == 1, np.random.choice([0, 1], size=300, p=[0.1, 0.9]),  # 90% chance of 1 for males
                         np.random.choice([0, 1], size=300, p=[0.9, 0.1])  # 90% chance of 0 for females
                        )

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a biased XGBoost model
biased_model = xgb.XGBClassifier(n_estimators=1000, max_depth=30, min_child_weight=10, random_state=42)
biased_model.fit(X_train, y_train)

# Evaluate the biased model
y_pred_biased = biased_model.predict(X_test)
confusion_biased = confusion_matrix(y_test, y_pred_biased)
print("Confusion Matrix for Biased Model:\n", confusion_biased)
print(classification_report(y_test, y_pred_biased))

# Debiasing Techniques
Reweightning(X_train, y_train, X_test, y_test, confusion_biased)

# Messaging Technique
Messaging(biased_model, X_test, y_test, confusion_biased)
