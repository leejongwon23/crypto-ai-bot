# meta_learning.py

import torch
import torch.nn as nn
import torch.optim as optim
import copy

class MAML:
    def __init__(self, model, inner_lr=0.01, outer_lr=0.001, inner_steps=1):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.optimizer = optim.Adam(self.model.parameters(), lr=outer_lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def adapt(self, X, y):
        # Clone model for inner loop adaptation
        adapted_model = copy.deepcopy(self.model)
        adapted_model.train()

        for _ in range(self.inner_steps):
            logits = adapted_model(X)
            loss = self.loss_fn(logits, y)
            grads = torch.autograd.grad(loss, adapted_model.parameters(), create_graph=True)
            for p, g in zip(adapted_model.parameters(), grads):
                p.data = p.data - self.inner_lr * g

        return adapted_model

    def meta_update(self, tasks):
        meta_loss = 0.0
        for X_train, y_train, X_val, y_val in tasks:
            adapted_model = self.adapt(X_train, y_train)
            adapted_model.eval()
            logits = adapted_model(X_val)
            loss = self.loss_fn(logits, y_val)
            meta_loss += loss

        meta_loss /= len(tasks)

        self.optimizer.zero_grad()
        meta_loss.backward()
        self.optimizer.step()

        return meta_loss.item()
