"""
Train a OneHotFullyConnected model on pads data
"""

import torch
from torch import nn
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
import numpy as np
from typing import List, Tuple
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

from pads_ml.model import OneHotFullyConnected

class OneHotFullyConnectedTrainer:

    def __init__(
        self,
        train_features: List[Path],
        valid_features: List[Path],
        train_labels: List[Path],
        valid_labels: List[Path],
    ):
        self.batch_size = 256
        self.train_features = train_features
        self.valid_features = valid_features
        self.train_labels = train_labels
        self.valid_labels = valid_labels
        self.train_loader, self.valid_loader = self.load_data()
        self.n_epoch = 2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {self.device}")
        self.model = OneHotFullyConnected()


    def train(self):
        """
        Train the model!
        """
        self.model.train()
        optimizer = AdamW(self.model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        logger.info("Starting epoch loop")
        for epoch in range(self.n_epoch):
            logger.info(f"Epoch {epoch}")
            for features, labels in self.train_loader:
                batch_size = features.shape[0]
                features, labels = features.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                output = self.model(features)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                if batch_size != self.batch_size:
                    self.evaluate()
            # logger.info(f"Training loss (last): {loss.item()}")
            # self.evaluate()
        self.model.eval()

    def evaluate(self):
        """
        Evaluate the model
        """
        self.model.eval()
        criterion = nn.BCELoss()
        total_loss, n_loss = 0, 0
        for features, labels in self.valid_loader:
            features, labels = features.to(self.device), labels.to(self.device)
            output = self.model(features)
            loss = criterion(output, labels)
            total_loss += loss.item()
            n_loss += 1
        avg_loss = total_loss / n_loss
        logger.info(f"Validation loss: {avg_loss}")

    def load_data(self) -> Tuple[DataLoader, DataLoader]:
        """
        Load data into a DataLoader
        """
        logger.info(f"Creating DataLoader ...")
        trainset = OneHotDataset(self.train_features, self.train_labels, self.batch_size)
        validset = OneHotDataset(self.valid_features, self.valid_labels, self.batch_size)
        trainloader = DataLoader(trainset, batch_size=None)
        validloader = DataLoader(validset, batch_size=None)
        return trainloader, validloader

    def save(self, path: Path) -> None:
        """
        Save the model
        """
        torch.save(self.model.state_dict(), path)

class OneHotDataset(IterableDataset):

    def __init__(
            self,
            feature_paths: List[Path],
            label_paths: List[Path],
            batch_size: int,
        ):
        super().__init__()

        self.feature_paths = sorted(feature_paths)
        self.label_paths = sorted(label_paths)
        self.batch_size = batch_size

    def __iter__(self):

        for fpath, lpath in zip(self.feature_paths, self.label_paths):

            # logger.info(f"Loading {fpath} and {lpath}")
            features = np.load(fpath)
            labels = np.load(lpath)

            num = features.shape[0]
            indices = np.arange(num)
            np.random.shuffle(indices)

            for start in range(0, num, self.batch_size):

                end = min(start + self.batch_size, num)

                batch_feature = features[indices[start:end]]
                batch_label = labels[indices[start:end]]
                yield (
                    torch.tensor(batch_feature, dtype=torch.float32),
                    torch.tensor(batch_label, dtype=torch.float32),
                )

