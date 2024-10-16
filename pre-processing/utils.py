import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

def generate_data():
    # Set the random seed for reproducibility
    np.random.seed(101)  # Ensures that the random numbers generated are the same each time

    # Define the number of samples
    num_samples = 300

    # Generate age: Normal distribution around 40 with a standard deviation of 10
    age = np.clip(np.random.normal(loc=40, scale=10, size=num_samples), 18, 70).astype(int)

    # Generate credit amounts: Log-normal distribution to simulate realistic credit amounts
    credit_amount = np.random.lognormal(mean=8, sigma=1, size=num_samples).astype(int)

    # Generate sex: Control the proportion (e.g., 60% male, 40% female)
    sex = np.random.choice([0, 1], size=num_samples, p=[0.4, 0.6])  # 0: female, 1: male

    # Generate income: Normal distribution around 50k with a standard deviation of 15k
    income = np.clip(np.random.normal(loc=50000, scale=15000, size=num_samples), 20000, 100000).astype(int)

    # Generate education level: Categorical variable with 3 levels
    education_levels = ['High School', 'Bachelor', 'Master']
    education = np.random.choice(education_levels, size=num_samples)

    # Generate marital status: Categorical variable
    marital_status = np.random.choice(['Single', 'Married', 'Divorced'], size=num_samples)

    # Generate employment status: Categorical variable
    employment_status = np.random.choice(['Employed', 'Unemployed', 'Student'], size=num_samples)

    # Generate loan purpose: Categorical variable
    loan_purposes = ['Home Improvement', 'Debt Consolidation', 'Personal Use', 'Education']
    loan_purpose = np.random.choice(loan_purposes, size=num_samples)

    # Generate previous defaults: Binary variable (0 or 1)
    previous_defaults = np.random.choice([0, 1], size=num_samples, p=[0.85, 0.15])  # 15% chance of default

    # Create biased labels based on sex, income, and previous defaults
    label = np.where(sex == 1, 
                    np.random.choice([0, 1], size=num_samples, p=[0.3, 0.7]),  # Males have a 70% chance of being 1
                    np.random.choice([0, 1], size=num_samples, p=[0.7, 0.3])   # Females have a 30% chance of being 1
                    )

    # Combine into a DataFrame
    data = pd.DataFrame({
        'age': age,
        'credit_amount': credit_amount,
        'sex': sex,  # 0: female, 1: male
        'label': label,  # Binary labels
        'income': income,
        'education': education,
        'marital_status': marital_status,
        'employment_status': employment_status,
        'loan_purpose': loan_purpose,
        'previous_defaults': previous_defaults,
    })

    return data

def Reweightning(X_train, y_train, X_test, y_test, confusion_biased):
    # Re-weighting: Assign weights to samples based on their group
    weights = np.where(X_train['sex'] == 1, 1.5, 1)  # Give higher weight to female samples

    # Train a debiased XGBoost model with sample weights
    debiased_model = xgb.XGBClassifier(n_estimators=1000, max_depth=30, min_child_weight=10, random_state=42)
    debiased_model.fit(X_train, y_train, sample_weight=weights)

    # Evaluate the debiased model
    y_pred_debiased = debiased_model.predict(X_test)
    confusion_debiased = confusion_matrix(y_test, y_pred_debiased)
    print("Confusion Matrix for Debiased Model:\n", confusion_debiased)
    print(classification_report(y_test, y_pred_debiased))

    # Plotting confusion matrices
    disp_biased = ConfusionMatrixDisplay(confusion_biased, display_labels=[0, 1])
    disp_biased.plot(cmap='Blues')
    disp_biased.ax_.set_title('Biased Model Confusion Matrix')

    disp_debiased = ConfusionMatrixDisplay(confusion_debiased, display_labels=[0, 1])
    disp_debiased.plot(cmap='Reds')
    disp_debiased.ax_.set_title('Debiased Model Confusion Matrix')

    # Show plots
    plt.show()

def apply_messaging(model, X):
    # Get probabilities of the positive class
    probabilities = model.predict_proba(X)[:, 1]
    adjusted_predictions = np.copy(probabilities)

    # Apply messaging based on 'sex' feature
    for i, sex in enumerate(X['sex']):
        if sex == 0:  # Female
            if probabilities[i] < 0.5:  # If predicted probability is low
                adjusted_predictions[i] += 0.1  # Increase probability
        else:  # Male
            if probabilities[i] < 0.5:  # If predicted probability is low
                adjusted_predictions[i] += 0.05  # Increase probability, but less for males

    # Normalize probabilities
    adjusted_predictions = np.clip(adjusted_predictions, 0, 1)  # Ensure probabilities are between 0 and 1
    return (adjusted_predictions >= 0.5).astype(int)  # Convert probabilities to binary predictions

def Messaging(biased_model, X_test, y_test, confusion_biased):
    # Apply messaging on biased model predictions
    y_pred_messaging = apply_messaging(biased_model, X_test)
    confusion_messaging = confusion_matrix(y_test, y_pred_messaging)
    print("Confusion Matrix after Messaging Technique:\n", confusion_messaging)
    print(classification_report(y_test, y_pred_messaging))

    # Plotting confusion matrices
    disp_biased = ConfusionMatrixDisplay(confusion_biased, display_labels=[0, 1])
    disp_biased.plot(cmap='Blues')
    disp_biased.ax_.set_title('Biased Model Confusion Matrix')

    disp_messaging = ConfusionMatrixDisplay(confusion_messaging, display_labels=[0, 1])
    disp_messaging.plot(cmap='Greens')
    disp_messaging.ax_.set_title('Debiased Model with Messaging Technique')