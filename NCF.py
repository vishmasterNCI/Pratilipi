from torch.autograd import Variable
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class NCF(pl.LightningModule):
    """ Neural Collaborative Filtering (NCF)

        Args:
            num_users (int): Number of unique users
            num_items (int): Number of unique items
            read_percent (pd.DataFrame): Dataframe containing the pratilipi read_percent for training
            all_pratilipiIds (list): List containing all pratilipiIds (train + test)
    """

    def __init__(self, num_users, num_items, read_percent, all_pratilipiIds):
        super().__init__()
        self.user_embedding = nn.Embedding(num_embeddings=num_users+1, embedding_dim=8)
        self.item_embedding = nn.Embedding(num_embeddings=num_items+1, embedding_dim=8)
        self.fc1 = nn.Linear(in_features=16, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.output = nn.Linear(in_features=32, out_features=1)
        self.read_percent = read_percent
        self.all_pratilipiIds = all_pratilipiIds

    def forward(self, user_input, item_input):

        # Pass through embedding layers
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)
        # Concat the two embedding layers
        vector = torch.cat([user_embedded, item_embedded], dim=-1)

        # Pass through dense layer
        vector = nn.ReLU()(self.fc1(vector))
        vector = nn.ReLU()(self.fc2(vector))

        # Output layer
        pred = nn.Sigmoid()(self.output(vector))

        return pred

    def training_step(self, batch, batch_idx):
        user_input, item_input, labels = batch
        predicted_labels = self(user_input, item_input)
        loss = nn.BCELoss()(predicted_labels, labels.view(-1, 1).float())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def train_dataloader(self):
        return DataLoader(PratilipiTrainDataset(self.read_percent, self.all_pratilipiIds),
                          batch_size=2048, num_workers=4)
