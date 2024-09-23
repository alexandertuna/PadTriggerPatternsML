"""
Train a OneHotFullyConnected model on pads data
"""

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
import pandas as pd
import numpy as np
from typing import Tuple

from .model import OneHotFullyConnected
from . import constants

class OneHotFullyConnectedTrainer:

    def __init__(
        self,
        features: np.array,
        labels: np.array,
    ):
        self.batch_size = 128
        # self.df = self.combine_inputs(signal, noise)
        # self.data, self.labels = self.convert_to_one_hot(self.df)
        self.features = features
        self.labels = labels
        self.loader = self.load_data()
        self.n_epoch = 3
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        print(f"Features:", self.features.shape)
        print(f"Labels:", self.labels.shape)
        # self.dataloader = OneHotFullyConnectedDataLoader()
        self.model = OneHotFullyConnected()

    def train(self):
        """
        Train the model!
        """
        self.model.train()
        optimizer = AdamW(self.model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        print("Starting epoch loop ...")
        for epoch in range(self.n_epoch):
            total_loss, n_loss = 0, 0
            for features, labels in self.loader:
                features, labels = features.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                output = self.model(features)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_loss += 1
            avg_loss = total_loss / n_loss
            print(f"Epoch {epoch}, loss: {avg_loss}")
        self.model.eval()

    def combine_inputs(
            self,
            signal: pd.DataFrame,
            noise: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Drop signal events with missing pads,
        and truncate noise to the same length as signal
        """
        print(f"Combining inputs ...")
        signal["label"] = 1
        noise["label"] = 0
        cols = [f"pad_{i}" for i in range(constants.LAYERS)] + ["label"]
        signal = signal[ (signal[cols] != -1).all(axis=1) ]
        noise = noise.iloc[ : len(signal) ]
        return pd.concat([signal[cols], noise], ignore_index=True)

    
    def load_data(self) -> DataLoader:
        """
        Load data into a DataLoader
        """
        print(f"Creating DataLoader ...")
        features = torch.tensor(self.features, dtype=torch.float32)
        labels = torch.tensor(self.labels, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(features, labels)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)


class OneHotDataset(Dataset):
    pass
