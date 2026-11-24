from __future__ import annotations

from dataclasses import dataclass
from math import inf


@dataclass
class EarlyStopping:
    patience: int
    minimum_improvement: float = 1e-4
    best_loss: float = inf
    stale_epochs: int = 0

    def update(self, loss: float) -> bool:
        if loss < self.best_loss - self.minimum_improvement:
            self.best_loss = loss
            self.stale_epochs = 0
            return False
        self.stale_epochs += 1
        return self.patience > 0 and self.stale_epochs >= self.patience
