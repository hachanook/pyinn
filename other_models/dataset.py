from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torch

class HeatDataset(Dataset):
    def __init__(self, csv_path, ROOT='/mnt/a/jgz1751/icml_parse_dataset/', dataset_type='train_full', clean_noise_type='clean', is_for_test=False):
        self.data = pd.read_csv(csv_path)
        
        if clean_noise_type == 'clean':
            output_label = 'u'
        else:
            output_label = 'u_noise'
         
        if dataset_type == 'train_full':
            train_data = pd.read_csv(ROOT + f'train_{clean_noise_type}.csv')
        elif dataset_type == 'train_10':
            train_data = pd.read_csv(ROOT + f'train_10_{clean_noise_type}.csv')
        elif dataset_type == 'train_30':
            train_data = pd.read_csv(ROOT + f'train_30_{clean_noise_type}.csv')
        else:
            raise ValueError(f"Invalid dataset_type: {dataset_type}")
        
        self.input_mins = train_data[['x', 'y', 't', 'k', 'p']].min()
        self.input_maxs = train_data[['x', 'y', 't', 'k', 'p']].max()

        # Compute min and max for output 'u'
        self.u_min = train_data[output_label].min()
        self.u_max = train_data[output_label].max()

        # Normalize input features
        self.inputs = self.normalize_inputs(self.data[['x', 'y', 't', 'k', 'p']].values.astype('float32'))

        # Normalize output target variable
        if is_for_test:
            output_label = 'u'
           
        self.outputs = self.normalize_u(self.data[[output_label]].values.astype('float32'))
    
    def normalize_inputs(self, inputs):
        return (inputs - self.input_mins.values) / (self.input_maxs.values - self.input_mins.values)

    def normalize_u(self, u):
        return (u - self.u_min) / (self.u_max - self.u_min)

    def denormalize_u(self, u):
        return u * (self.u_max - self.u_min) + self.u_min

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx], dtype=torch.float32), torch.tensor(self.outputs[idx], dtype=torch.float32)


if __name__ == '__main__':
    dataset = HeatDataset('/mnt/a/jgz1751/icml_parse_dataset/train_noisy.csv', dataset_type='train_full', clean_noise_type='noisy')
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    for inputs, outputs in train_loader:
        print(f"inputs: {inputs.shape}, outputs: {outputs.shape}")
        break