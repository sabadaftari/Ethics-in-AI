import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from utils import generate_data, plot
from model import LoanApprovalModel, LearnableFairnessRegularizer, train, eval

torch.manual_seed(42)

# Generate data
df = generate_data()

# Split features and target variable
X = df.drop(columns=['loan_decision'])
y = df['loan_decision']
print(y.value_counts())
sensitive_features = df[['gender', 'race']]  # Example sensitive features

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(X, y, sensitive_features, 
                                                                                         test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train.values)
y_train_tensor = torch.FloatTensor(y_train.values).view(-1)  # Reshape for binary classification
sensitive_features_train = torch.FloatTensor(sensitive_train.values)

X_test_tensor = torch.FloatTensor(X_test.values)
y_test_tensor = torch.FloatTensor(y_test.values).view(-1)  # Reshape for binary classification
 
# Initialize models
input_size = X_train_tensor.shape[1]
biased_model = LoanApprovalModel(input_size)
unbiased_model = LoanApprovalModel(input_size)

# Hyperparameters
epochs = 200
fairness_lambda = 0.01  # Regularization strength for fairness
precision_biased_list = []
recall_biased_list = []
precision_unbiased_list = []
recall_unbiased_list = []
# Lists to hold loss values
biased_losses = []
unbiased_losses = []

# Initialize fairness regularizer
fairness_regularizer = LearnableFairnessRegularizer()
optimizer_biased = optim.Adam(biased_model.parameters(), lr=0.0001)  # Decrease the learning rate
optimizer_unbiased = optim.Adam(list(unbiased_model.parameters()) + list(fairness_regularizer.parameters()), lr=0.0001)

train(epochs, biased_model, unbiased_model,
          X_train_tensor,y_train_tensor,
          fairness_regularizer, sensitive_features_train,
          fairness_lambda, optimizer_biased,optimizer_unbiased,
          biased_losses, unbiased_losses,precision_biased_list,
          recall_biased_list, precision_unbiased_list, recall_unbiased_list)

eval(biased_model,unbiased_model, X_test_tensor, y_test)

# Plotting the loss curves
plot(epochs, biased_losses, unbiased_losses,
         precision_biased_list, precision_unbiased_list, 
         recall_biased_list, recall_unbiased_list)