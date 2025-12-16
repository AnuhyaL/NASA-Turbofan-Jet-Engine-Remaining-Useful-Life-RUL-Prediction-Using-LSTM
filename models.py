import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMRegressor(nn.Module):
    """
    A simple LSTM-based regressor for RUL prediction.
    """

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)          # out: (batch, seq_len, hidden_size)
        last_hidden = out[:, -1, :]    # take last time step: (batch, hidden_size)
        pred = self.fc(last_hidden)    # (batch, 1)
        return pred.squeeze(-1)        # (batch,)


class RULPredictor:
    """
    Wrapper around an LSTMRegressor to provide fit() and predict()
    using numpy arrays for convenience.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        lr: float = 1e-3,
        device: str | None = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = LSTMRegressor(input_size, hidden_size, num_layers, dropout).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 10, batch_size: int = 64):
        """
        Train the LSTM model.

        Parameters
        ----------
        X : np.ndarray
            Shape (num_samples, seq_len, num_features)
        y : np.ndarray
            Shape (num_samples,)
        epochs : int
            Number of training epochs.
        batch_size : int
            Batch size for training.
        """
        self.model.train()

        dataset = TensorDataset(
            torch.from_numpy(X).float(),
            torch.from_numpy(y).float(),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()
                preds = self.model(batch_X)
                loss = self.criterion(preds, batch_y)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * batch_X.size(0)

            epoch_loss /= len(dataset)
            # You can print this in console; Streamlit will just ignore it.
            print(f"[Epoch {epoch+1}/{epochs}] Train MSE: {epoch_loss:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict RUL for given input sequences.

        Parameters
        ----------
        X : np.ndarray
            Shape (num_samples, seq_len, num_features)

        Returns
        -------
        np.ndarray
            Shape (num_samples,) predicted RUL values.
        """
        self.model.eval()
        X_tensor = torch.from_numpy(X).float().to(self.device)

        with torch.no_grad():
            preds = self.model(X_tensor).cpu().numpy()

        return preds

    def save(self, path: str):
        """
        Save the model weights to a file.
        """
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        """
        Load the model weights from a file.
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
