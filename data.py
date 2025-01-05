import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class BikeDemandDataset(Dataset):
    def __init__(self, file_path, data_type='train'):
        """
        Args:
            file_path (str): Path to the CSV file.
            data_type (str): Type of dataset - 'train' or 'validation'.
        """
        self.file_path = file_path
        self.data_type = data_type

        # Load the data
        data = pd.read_csv(file_path)

        # Separate input features and labels
        self.features = data.drop(columns=['increase_stock'])
        self.labels = data['increase_stock']

        # Encode the labels (e.g., 'low_bike_demand' -> 0, etc.)
        self.label_mapping = {label: idx for idx, label in enumerate(self.labels.unique())}
        self.labels = self.labels.map(self.label_mapping)

        # Split the data into train and validation sets
        train_features, val_features, train_labels, val_labels = train_test_split(
            self.features, self.labels, test_size=0.3, random_state=42
        )

        # Select the appropriate subset based on data_type
        if data_type == 'train':
            self.features = train_features
            self.labels = train_labels
        elif data_type == 'validation':
            self.features = val_features
            self.labels = val_labels
        else:
            raise ValueError("data_type must be either 'train' or 'validation'")

        # Convert data to tensors
        self.features = torch.tensor(self.features.values, dtype=torch.float32)
        self.labels = torch.tensor(self.labels.values, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
class BikeDemandDataset_v01(Dataset):
    def __init__(self, file_path, data_type='train'):
        """
        Args:
            file_path (str): Path to the CSV file.
            data_type (str): Type of dataset - 'train' or 'validation'.
        """
        self.file_path = file_path
        self.data_type = data_type

        # Load the data
        data = pd.read_csv(file_path)

        # Separate input features and labels
        self.features = data.drop(columns=['increase_stock'])
        self.features = pd.get_dummies(self.features, columns=["hour_of_day", "day_of_week", "month"])
        
        self.features['temp_cold'] = self.features['temp'] < 5
        self.features['dew_low'] = self.features['dew'] < -10
        self.features['precip_0'] = self.features['precip'] == 0
        self.features['snowdepth_0'] = self.features['snowdepth'] == 0

        self.features = self.features.astype(float)
        self.labels = data['increase_stock']

        # Encode the labels (e.g., 'low_bike_demand' -> 0, etc.)
        self.label_mapping = {label: idx for idx, label in enumerate(self.labels.unique())}
        self.labels = self.labels.map(self.label_mapping)

        # Split the data into train and validation sets
        train_features, val_features, train_labels, val_labels = train_test_split(
            self.features, self.labels, test_size=0.3, random_state=41
        )

        # Select the appropriate subset based on data_type
        if data_type == 'train':
            self.features = train_features
            self.labels = train_labels
        elif data_type == 'validation':
            self.features = val_features
            self.labels = val_labels
        else:
            raise ValueError("data_type must be either 'train' or 'validation'")

        # Convert data to tensors
        self.features = torch.tensor(self.features.values, dtype=torch.float32)
        self.labels = torch.tensor(self.labels.values, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]