import sys
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from scipy import stats

#%% Complex models

# Complex-Valued Linear Regression Model
class ComplexLinearBase(nn.Module):
    def __init__(self, input_dim):
        super(ComplexLinearBase, self).__init__()
        self.w = nn.Parameter(torch.randn(input_dim, dtype=torch.cfloat))
        self.b = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))  # Intercept term

    def forward(self, x):
        # y_pred = torch.matmul(x, self.w.conj()).real
        y_pred = (torch.matmul(x.real, self.w.real) + torch.matmul(x.imag, self.w.imag)) + self.b
        return y_pred

    def l1_regularisation(self):
        # L1 norm of complex weights: sum of magnitudes
        return torch.sum(torch.abs(self.w))

class ComplexLinear(nn.Module):
    def __init__(self, input_dim, lambda_reg=0.0, max_epochs=10000, lr=0.01, patience=500, device="cuda"):
        super(ComplexLinear, self).__init__()
        self.input_dim = input_dim
        self.lambda_reg = lambda_reg
        self.max_epochs = max_epochs
        self.patience = patience  
        self.device = torch.device(device)
        self.model = ComplexLinearBase(input_dim).to(self.device)
        self.optimiser = optim.Adam(self.model.parameters(), lr=lr)
        self.task = None
        self.coef_ = None
        self.intercept_ = None

    def process_input(self, X):
        return torch.from_numpy(X.to_numpy()).to(self.device)

    def fit_regressor(self, X, y, val_split=0.3):
        self.task = 'regression'
        X, y = self.process_input(X), self.process_input(y)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_split, shuffle=True)
        
        criterion = nn.MSELoss()
        best_loss = float('inf')
        epochs_no_improve = 0
        for epoch in tqdm(range(self.max_epochs)):
            self.model.train()
            self.optimiser.zero_grad()
            y_pred = self.model(X_train)
            train_loss = criterion(y_pred, y_train)
            if self.lambda_reg > 0:
                train_loss += self.lambda_reg * self.model.l1_regularisation()
            train_loss.backward()
            self.optimiser.step()

            # Validation phase
            self.model.eval()
            with torch.no_grad():
                y_val_pred = self.model(X_val)
                val_loss = criterion(y_val_pred, y_val)
                if self.lambda_reg > 0:
                    val_loss += self.lambda_reg * self.model.l1_regularisation()

            # Check for improvement
            if val_loss < best_loss:
                best_loss = val_loss
                epochs_no_improve = 0
                best_model_state = self.model.state_dict()
            else:
                epochs_no_improve += 1

            # Early stopping
            if epochs_no_improve >= self.patience:
                print(f'Early stopping at epoch {epoch+1}')
                self.model.load_state_dict(best_model_state)
                break

            if (epoch+1) % 1000 == 0:
                print(f'Epoch [{epoch+1}/{self.max_epochs}], Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
        print('Final Validation Loss:', best_loss.item())
        self._save_parameters()

    def fit_classifier(self, X, y, val_split=0.3):
        self.task = 'classification'
        X, y = self.process_input(X), self.process_input(y)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_split, shuffle=True)
        
        is_binary = y.unique().size(0) == 2
        criterion = nn.BCEWithLogitsLoss() if is_binary else nn.CrossEntropyLoss()
        if not is_binary:
            sys.exit('Only binary classification is supported as of now!')
        best_acc = 0.0
        epochs_no_improve = 0
        for epoch in tqdm(range(self.max_epochs)):
            self.model.train()
            self.optimiser.zero_grad()
            logits = self.model(X_train)
            train_loss = criterion(logits, y_train)
            if self.lambda_reg > 0:
                train_loss += self.lambda_reg * self.model.l1_regularisation()
            train_loss.backward()
            self.optimiser.step()

            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(X_val)
                val_loss = criterion(val_logits, y_val)
                if self.lambda_reg > 0:
                    val_loss += self.lambda_reg * self.model.l1_regularisation()
                val_predictions = (torch.sigmoid(val_logits) > 0.5).float()
                val_accuracy = (val_predictions == y_val).float().mean()

            # Check for improvement
            if val_accuracy > best_acc:
                best_acc = val_accuracy
                epochs_no_improve = 0
                best_model_state = self.model.state_dict()
            else:
                epochs_no_improve += 1

            # Early stopping
            if epochs_no_improve >= self.patience:
                print(f'Early stopping at epoch {epoch+1}')
                self.model.load_state_dict(best_model_state)
                break

            if (epoch+1) % 1000 == 0:
                print(f'Epoch [{epoch+1}/{self.max_epochs}], Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Accuracy: {val_accuracy.item():.4f}')
        print('Final Validation Accuracy:', best_acc.item())
        self._save_parameters()

    def predict(self, X_test):
        if self.task is None:
            raise ValueError("Model has not been trained yet. Call fit_regressor or fit_classifier first.")
        X_test = self.process_input(X_test)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_test)
            if self.task == 'regression':
                predictions = outputs  # For regression, output the raw predictions
            elif self.task == 'classification':
                probabilities = torch.sigmoid(outputs)
                predictions = (probabilities > 0.5).float()  # For binary classification
            else:
                raise ValueError(f"Unknown task: {self.task}")
        return predictions.cpu().numpy()

    def _save_parameters(self):
        with torch.no_grad():
            self.coef_ = self.model.w.real.cpu().numpy() + 1j * self.model.w.imag.cpu().numpy()  
            self.intercept_ = self.model.b.cpu().item()

# # Set random seed for reproducibility
# torch.manual_seed(0)

# # Assume we have N samples with complex-valued latent vectors of dimension D
# N = 100  # Number of samples
# D = 50   # Dimension of latent space

# # Generate random complex-valued latent vectors
# X_real = torch.randn(N, D)
# X_imag = torch.randn(N, D)
# X = X_real + 1j * X_imag  # Complex-valued inputs

# # Generate real-valued phenotypes (for regression)
# w_true_real = torch.randn(D)
# w_true_imag = torch.randn(D)
# w_true = w_true_real + 1j * w_true_imag
# y_regression = torch.randn(N) + 0.1 * torch.randn(N)  # Add noise

# # Generate binary disease labels (for classification)
# logits = torch.randn(N)
# probabilities = torch.sigmoid(logits)
# y_classification = (probabilities > 0.5).float()

# # Instantiate and train the regression model
# regression_model = ComplexLinear(D, lambda_reg=0.1)
# regression_model.fit_regressor(X, y_regression)

# # # Instantiate and train the classification model
# # classification_model = ComplexLinear(D, lambda_reg=0.1)
# # classification_model.fit_classifier(X, y_classification)

# coefficients = pd.DataFrame(regression_model.coef_, columns=['Coefficient'])
# intercept = regression_model.intercept_

# dof = N - D - 1
# residuals = y_regression - regression_model.predict(X)
# mse = np.sum(residuals.cpu().numpy()**2) / dof

# X = X.cpu().numpy()
# t_values, p_values = [], []
# for i, coef in enumerate(coefficients['Coefficient']):
#     standard_error = np.sqrt(mse / np.sum((X[:, i] - X[:, i].mean())**2))
#     t_stat = coef / standard_error
#     p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), dof))
#     t_values.append(t_stat)
#     p_values.append(p_value)

# print('Coefficients:', coefficients)
# print('Intercept:', intercept)
# print('T-values:', t_values)
# print('P-values:', p_values)