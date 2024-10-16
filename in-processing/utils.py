
import numpy as np
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt

def generate_data():
    np.random.seed(42)
    

    # Sample data creation
    num_samples = 1000
    data = {
        'age': np.random.randint(18, 70, num_samples),
        'gender': np.random.randint(0, 2, num_samples),  # 0: Male, 1: Female
        'race': np.random.randint(0, 4, num_samples),    # 0: Race1, 1: Race2, 2: Race3, 3: Race4
        'income': np.random.randint(20000, 120000, num_samples),
        'credit_score': np.random.randint(300, 850, num_samples),
        'loan_amount': np.random.randint(1000, 50000, num_samples),
        'loan_term': np.random.randint(1, 30, num_samples),  # Years
        'employment_status': np.random.randint(0, 2, num_samples),  # 0: Unemployed, 1: Employed
        'marital_status': np.random.randint(0, 2, num_samples),  # 0: Single, 1: Married
        'number_of_dependents': np.random.randint(0, 5, num_samples),
        'home_ownership': np.random.randint(0, 2, num_samples),  # 0: Renter, 1: Owner
        'education_level': np.random.choice(['High School', 'Bachelors', 'Masters', 'PhD'], num_samples),
        'debt_to_income_ratio': np.random.uniform(0, 1, num_samples),
        'bank_balance': np.random.randint(0, 50000, num_samples),
        'previous_loans': np.random.randint(0, 10, num_samples),
        'default_history': np.random.randint(0, 2, num_samples),
        'num_accounts': np.random.randint(1, 10, num_samples),
        'loan_purpose': np.random.randint(0, 3, num_samples),  # 0: Personal, 1: Business, 2: Education
        'self_employed': np.random.randint(0, 2, num_samples),  # 0: No, 1: Yes
        'credit_card_debt': np.random.randint(0, 15000, num_samples),
        'savings_account_balance': np.random.randint(0, 50000, num_samples),
        'age_of_credit_history': np.random.randint(0, 30, num_samples),
        'rental_history': np.random.randint(0, 5, num_samples),
        'payment_history': np.random.randint(0, 3, num_samples),  # 0: Poor, 1: Average, 2: Good
        'bankruptcy_history': np.random.randint(0, 2, num_samples),  # 0: No, 1: Yes
        'business_experience': np.random.randint(0, 10, num_samples),
        'social_media_score': np.random.uniform(0, 100, num_samples),
        'loan_decision': np.random.randint(0, 2, num_samples)  # 0: Denied, 1: Approved
    }

    df = pd.DataFrame(data)

    # Introduce bias in the loan_decision
    # Let's create a very biased dataset with 90% approved and 10% denied
    num_approved = int(num_samples * 0.7)  # 90% approvals
    num_denied = num_samples - num_approved  # 10% denials

    # Create biased loan_decision
    loan_decision = np.array([1] * num_approved + [0] * num_denied)

    # Shuffle the loan_decision array
    np.random.shuffle(loan_decision)
    df['loan_decision'] = loan_decision

    # Convert categorical columns to numeric
    df['gender'] = df['gender'].astype(int)
    df['race'] = df['race'].astype(int)
    df['employment_status'] = df['employment_status'].astype(int)
    df['marital_status'] = df['marital_status'].astype(int)
    df['home_ownership'] = df['home_ownership'].astype(int)
    df['loan_purpose'] = df['loan_purpose'].astype(int)

    # One-hot encode the 'education_level' column
    df = pd.get_dummies(df, columns=['education_level'], drop_first=True)

    # Convert Boolean columns to integers (0 and 1)
    bool_columns = df.select_dtypes(include=['bool']).columns
    df[bool_columns] = df[bool_columns].astype(int)

    return df

# Loss Function that integrates the fairness regularizer
def loss_function(regularizer, predictions, true_labels, sensitive_features, alpha=0.01):
    # Calculate the main task loss (binary cross-entropy)
    main_loss = F.binary_cross_entropy_with_logits(predictions, true_labels)

    # Calculate the fairness regularizer loss
    fairness_loss = regularizer(predictions, sensitive_features)

    # Combine both losses with a weighting factor alpha
    total_loss = main_loss + alpha * fairness_loss
    return total_loss

def plot(epochs, biased_losses, unbiased_losses,
         precision_biased_list, precision_unbiased_list, 
         recall_biased_list, recall_unbiased_list):
    # Plotting the loss curves
    plt.figure(figsize=(12, 6))
    plt.plot(biased_losses, label='Biased Model Loss', color='blue')
    plt.plot(unbiased_losses, label='Unbiased Model Loss', color='orange')

    # Add titles and labels
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

    # Plotting Precision and Recall
    plt.figure(figsize=(12, 6))

    # Precision Plot
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), precision_biased_list, label='Biased Model Precision', color='red')
    plt.plot(range(epochs), precision_unbiased_list, label='Unbiased Model Precision', color='blue')
    plt.title('Precision throughout Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.legend()

    # Recall Plot
    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), recall_biased_list, label='Biased Model Recall', color='red')
    plt.plot(range(epochs), recall_unbiased_list, label='Unbiased Model Recall', color='blue')
    plt.title('Recall throughout Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.legend()

    plt.tight_layout()
    plt.show()