import torch
from torch import nn
from torchcomplex import nn as cnn
from sklearn.model_selection import KFold

class Lasso(nn.Module):
    def __init__(self, in_features, alpha=1.0, max_iter=1000, tol=1e-4):
        super(Lasso, self).__init__()
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.linear = nn.Linear(in_features, 1, bias=False)

    def forward(self, X):
        return self.linear(X)

    def fit(self, X, y):
        optimiser = torch.optim.SGD(self.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()

        w_prev = torch.tensor(0.).to(X.device)
        for _ in range(self.max_iter):
            optimiser.zero_grad()
            y_pred = self(X)
            loss = loss_fn(y_pred, y) + self.alpha * torch.norm(self.linear.weight, 1)
            loss.backward()
            optimiser.step()
            
            w = self.linear.weight.detach()
            if bool(torch.norm(w_prev - w) < self.tol):
                break
            w_prev = w

        return self
    
    def predict(self, X):
        if X is not torch.Tensor:
            X = torch.tensor(X).to("cuda")
        return self(X)


class LassoCV(nn.Module):
    def __init__(self, eps=1e-3, n_alphas=100, max_iter=1000, tol=1e-4, cv=5):
        super(LassoCV, self).__init__()
        self.eps = eps
        self.n_alphas = n_alphas
        self.max_iter = max_iter
        self.tol = tol
        self.cv = cv
        self.alpha_ = None
        self.coef_ = None

    def fit(self, X, y):
        if X is not torch.Tensor:
            X = torch.tensor(X).to("cuda")
        if y is not torch.Tensor:
            y = torch.tensor(y).unsqueeze(-1).to("cuda")

        alpha_max = torch.max(torch.abs(X.T.matmul(y))) / X.shape[0]
        alphas = torch.logspace(torch.log10(alpha_max * self.eps), torch.log10(alpha_max), self.n_alphas)

        kfold = KFold(n_splits=self.cv)
        cv_errors = torch.zeros(self.n_alphas)

        for alpha in alphas:
            errors = []
            for train_idx, val_idx in kfold.split(X):
                model = Lasso(in_features=X.shape[-1], alpha=alpha.item(), max_iter=self.max_iter, tol=self.tol).to("cuda")
                model.fit(X[train_idx], y[train_idx])
                y_val_pred = model.forward(X[val_idx])
                error = torch.mean((y[val_idx] - y_val_pred) ** 2)
                errors.append(error.item())
            cv_errors[alphas == alpha] = torch.tensor(errors).mean()

        self.alpha_ = alphas[torch.argmin(cv_errors)].item()
        self.model = Lasso(in_features=X.shape[-1], alpha=self.alpha_, max_iter=self.max_iter, tol=self.tol).to("cuda").fit(X, y)
        self.coef_ = self.model.linear.weight.data
        self.predict = self.model.predict

        return self