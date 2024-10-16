import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import loss_function
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# Define the neural network model
class LoanApprovalModel(nn.Module):
    def __init__(self, input_size):
        super(LoanApprovalModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)  # Batch Normalization after first layer
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)   # Batch Normalization after second layer
        self.fc3 = nn.Linear(256, 32)
        self.bn3 = nn.BatchNorm1d(32)   # Batch Normalization after third layer
        self.fc4 = nn.Linear(32, 1)
        self.S = nn.Sigmoid()

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))  # Apply Batch Norm after ReLU
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        return self.fc4(x)
    
# Learnable Fairness Regularizer Class
class LearnableFairnessRegularizer(nn.Module):
    def __init__(self, fairness_metric="demographic_parity", temp=1.0):
        super(LearnableFairnessRegularizer, self).__init__()
        # Learnable weights for the regularizer with constraints
        self.weight_group_0 = nn.Parameter(torch.tensor(0.5, requires_grad=True))
        self.weight_group_1 = nn.Parameter(torch.tensor(0.5, requires_grad=True))
        self.fairness_metric = fairness_metric  # Allow multiple fairness metrics
        self.temp = temp  # Temperature scaling for sensitivity

    def forward(self, predictions, sensitive_features, true_labels=None):
        # Apply sigmoid with temperature scaling to obtain probabilities
        positive_rate = torch.sigmoid(predictions / self.temp)

        # Split the predictions by sensitive feature group (binary feature assumed)
        group_0_mask = (sensitive_features[:, 0] == 0)
        group_1_mask = (sensitive_features[:, 0] == 1)

        group_0_positive = positive_rate[group_0_mask].mean()
        group_1_positive = positive_rate[group_1_mask].mean()

        # Calculate demographic parity difference (or other fairness metrics)
        if self.fairness_metric == "demographic_parity":
            # Weighted demographic parity
            weighted_group_0 = self.weight_group_0 * group_0_positive
            weighted_group_1 = self.weight_group_1 * group_1_positive
            fairness_penalty = (weighted_group_0 - weighted_group_1).pow(2)

        elif self.fairness_metric == "equal_opportunity" and true_labels is not None:
            # True positive rates for equal opportunity
            true_positive_rate_group_0 = ((predictions.round() == 1) & (true_labels == 1))[group_0_mask].float().mean()
            true_positive_rate_group_1 = ((predictions.round() == 1) & (true_labels == 1))[group_1_mask].float().mean()
            weighted_group_0 = self.weight_group_0 * true_positive_rate_group_0
            weighted_group_1 = self.weight_group_1 * true_positive_rate_group_1
            fairness_penalty = (weighted_group_0 - weighted_group_1).pow(2)

        else:
            raise ValueError(f"Unsupported fairness metric: {self.fairness_metric}")

        # Add constraints on the weights to prevent instability (soft constraints)
        weight_constraint = (self.weight_group_0 - 0.5).pow(2) + (self.weight_group_1 - 0.5).pow(2)

        # Total loss combines fairness penalty with the weight regularization
        total_loss = fairness_penalty + 0.01 * weight_constraint

        return total_loss
    
def train(epochs, biased_model, unbiased_model,
          X_train_tensor,y_train_tensor,
          fairness_regularizer, sensitive_features_train,
          fairness_lambda, optimizer_biased,optimizer_unbiased,
          biased_losses, unbiased_losses,precision_biased_list,
          recall_biased_list, precision_unbiased_list, recall_unbiased_list):
    for epoch in range(epochs):
        biased_model.train()
        unbiased_model.train()

        # Forward pass
        y_pred_biased = biased_model(X_train_tensor).squeeze()  # Squeeze to match label dimensions
        y_pred_unbiased = unbiased_model(X_train_tensor).squeeze()

        # Compute loss for biased model (no fairness regularization)
        biased_loss = F.binary_cross_entropy_with_logits(y_pred_biased, y_train_tensor)

        # Compute loss for unbiased model with fairness regularization
        unbiased_loss_with_fairness = loss_function(
            regularizer=fairness_regularizer,
            predictions=y_pred_unbiased,
            true_labels=y_train_tensor,
            sensitive_features=sensitive_features_train,
            alpha=fairness_lambda  # Alpha is the weighting factor for fairness loss
        )

        # Backward pass and optimization (biased model)
        optimizer_biased.zero_grad()
        biased_loss.backward()
        optimizer_biased.step()

        # Backward pass and optimization (unbiased model)
        optimizer_unbiased.zero_grad()
        unbiased_loss_with_fairness.backward()
        optimizer_unbiased.step()
        
        # Print training status
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], '
                f'Biased Model Loss: {biased_loss.item():.4f}, '
                f'Unbiased Model Loss with Fairness: {unbiased_loss_with_fairness.item():.4f}')

        # Save loss values
        biased_losses.append(biased_loss.item())
        unbiased_losses.append(unbiased_loss_with_fairness.item())

        # Calculate metrics for biased model
        y_pred_biased_binary = (torch.sigmoid(y_pred_biased) > 0.5).float()
        precision_biased = precision_score(y_train_tensor.cpu().numpy(), y_pred_biased_binary.detach().cpu().numpy())
        recall_biased = recall_score(y_train_tensor.cpu().numpy(), y_pred_biased_binary.detach().cpu().numpy())
        precision_biased_list.append(precision_biased)
        recall_biased_list.append(recall_biased)

        # Calculate metrics for unbiased model
        y_pred_unbiased_binary = (torch.sigmoid(y_pred_unbiased) > 0.5).float()
        precision_unbiased = precision_score(y_train_tensor.cpu().numpy(), y_pred_unbiased_binary.detach().cpu().numpy())
        recall_unbiased = recall_score(y_train_tensor.cpu().numpy(), y_pred_unbiased_binary.detach().cpu().numpy())
        precision_unbiased_list.append(precision_unbiased)
        recall_unbiased_list.append(recall_unbiased)

def eval(biased_model,unbiased_model,
         X_test_tensor, y_test):
    # Evaluation
    biased_model.eval()
    unbiased_model.eval()
    # Make predictions
    with torch.no_grad():
        y_pred_biased = torch.sigmoid(biased_model(X_test_tensor)).squeeze().numpy()
        y_pred_unbiased = torch.sigmoid(unbiased_model(X_test_tensor)).squeeze().numpy()

    # Convert probabilities to binary outputs
    y_pred_biased_classes = (y_pred_biased > 0.5).astype(int)
    y_pred_unbiased_classes = (y_pred_unbiased > 0.5).astype(int)

    # Calculate accuracy
    accuracy1 = accuracy_score(y_test, y_pred_biased_classes)
    print(f'Biased Model Overall Accuracy: {accuracy1:.2f}')

    accuracy2 = accuracy_score(y_test, y_pred_unbiased_classes)
    print(f'Unbiased Model Overall Accuracy: {accuracy2:.2f}')

    # Print classification reports
    print("Biased Model Classification Report:")
    print(classification_report(y_test, y_pred_biased_classes))

    print("Unbiased Model Classification Report:")
    print(classification_report(y_test, y_pred_unbiased_classes))
