import torch.optim as optim

from torch.autograd import Variable
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class PratilipiTrainDataset(Dataset):
    """Pratilipi PyTorch Dataset for Training

    Args:
        read_percent (pd.DataFrame): Dataframe containing the pratilipi read_percent
        all_pratilipiIds (list): List containing all pratilipiIds

    """

    def __init__(self, read_percent, all_pratilipiIds):
        self.users, self.items, self.labels = self.get_dataset(read_percent, all_pratilipiIds)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

    def get_dataset(self, read_percent, all_pratilipiIds):
        users, items, labels = [], [], []
        user_item_set = set(zip(read_percent['user_id'], read_percent['pratilipi_id']))

        num_negatives = 4
        for u, i in user_item_set:
            users.append(u)
            items.append(i)
            labels.append(1)
            for _ in range(num_negatives):
                negative_item = np.random.choice(all_pratilipiIds)
                while (u, negative_item) in user_item_set:
                    negative_item = np.random.choice(all_pratilipiIds)
                users.append(u)
                items.append(negative_item)
                labels.append(0)

        return torch.tensor(users), torch.tensor(items), torch.tensor(labels)
