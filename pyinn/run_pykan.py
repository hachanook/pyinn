import argparse
import torch
from kan import KAN
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm

torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def get_args():
    parser = argparse.ArgumentParser(description='Train a PyTorch Lightning Model')
    parser.add_argument('--ROOT', type=str, default='/mnt/a/jgz1751/icml_parse_dataset/')
    parser.add_argument('--train_type', type=str, default='train')
    parser.add_argument('--clean_noise_type', type=str, default='clean', help='Clean or noisy data')
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--opt', type=str, default='LBFGS')
    parser.add_argument('--width', type=int, nargs='+', default=[5, 10, 5, 1])
    parser.add_argument('--grid', type=int, default=10)
    parser.add_argument('--gpu_id', type=int, default=5)

    return parser.parse_args()


class HeatDataset(Dataset):
    def __init__(self, csv_path, ROOT='/mnt/a/jgz1751/icml_parse_dataset/', dataset_type='train', clean_noise_type='clean', device='cpu', is_for_test=False):
        self.device = device
        self.data = pd.read_csv(csv_path)
        
        if clean_noise_type == 'clean':
            output_label = 'u'
        else:
            output_label = 'u_noise'
            
        if dataset_type == 'train':
            train_data = pd.read_csv(ROOT + f'train_{clean_noise_type}.csv')
        elif dataset_type == 'train_10':
            train_data = pd.read_csv(ROOT + f'train_10_{clean_noise_type}.csv')
        elif dataset_type == 'train_30':
            train_data = pd.read_csv(ROOT + f'train_30_{clean_noise_type}.csv')
        else:
            raise ValueError(f"Invalid dataset_type: {dataset_type}")
        
        self.input_mins = train_data[['x', 'y', 't', 'k', 'p']].min()
        self.input_maxs = train_data[['x', 'y', 't', 'k', 'p']].max()

        self.u_min = train_data[output_label].min()
        self.u_max = train_data[output_label].max()

        self.inputs = self.normalize_inputs(self.data[['x', 'y', 't', 'k', 'p']].values.astype('float32'))
        if is_for_test:
            output_label = 'u'
        self.outputs = self.normalize_u(self.data[[output_label]].values.astype('float32'))

        # Move data to device
        self.inputs = torch.tensor(self.inputs, dtype=torch.float32).to(self.device)
        self.outputs = torch.tensor(self.outputs, dtype=torch.float32).to(self.device)
    
    def normalize_inputs(self, inputs):
        return (inputs - self.input_mins.values) / (self.input_maxs.values - self.input_mins.values)

    def normalize_u(self, u):
        return (u - self.u_min) / (self.u_max - self.u_min)

    def denormalize_u(self, u):
        return u * (self.u_max - self.u_min) + self.u_min

    def get_dataset(self):
        return self.inputs, self.outputs
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]
    

def prepare_dataset(args, device, clean_noise_type='clean'):
    dataset = {}
    train_file = args.ROOT + f'{args.train_type}_{clean_noise_type}.csv'
    val_file = args.ROOT + f'val_{clean_noise_type}.csv'
    test_file = args.ROOT + f'test_{clean_noise_type}.csv'
    test_clean_file = args.ROOT + f'test_clean.csv'
    print(f"train_file: {train_file}, val_file: {val_file}, test_file: {test_file}")
    
    train_dataset = HeatDataset(train_file, dataset_type=args.train_type, clean_noise_type=clean_noise_type, device=device)
    test_dataset = HeatDataset(val_file, dataset_type=args.train_type, clean_noise_type=clean_noise_type, device=device)
    test_real_dataset = HeatDataset(test_file, dataset_type=args.train_type, clean_noise_type=clean_noise_type, device=device)
    test_real_clean_dataset = HeatDataset(test_clean_file, dataset_type=args.train_type, clean_noise_type=clean_noise_type, device=device, is_for_test=True)

    dataset['train_input'], dataset['train_label'] = train_dataset.get_dataset()
    dataset['test_input'], dataset['test_label'] = test_dataset.get_dataset()
    dataset['test_real_input'], dataset['test_real_label'] = test_real_dataset.get_dataset()

    return dataset, train_dataset, test_dataset, test_real_dataset, test_real_clean_dataset


def evaluate(loader, dataset, model, device):
    model.to(device)  # Ensure model is on the correct device
    model.eval()
    mse_total = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(loader):
            inputs, targets = inputs.to(device), targets.to(device)  # Move batch to device
            preds_each = model.forward(inputs)
            mse_total += torch.sum((preds_each - targets) ** 2).item()
            total_samples += len(targets)
            
    rmse = (mse_total / total_samples) ** 0.5
    return rmse


if __name__ == '__main__':
    args = get_args()
    print("Arguments:")
    for key, value in vars(args).items():
        print(f"-----{key}: {value}")
    
    # Set the device to GPU
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(f'cuda:{args.gpu_id}')
    print(f"Using GPU {args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)}")

    dataset, train_dataset, test_dataset, test_real_dataset, test_real_clean_dataset = prepare_dataset(args, device, args.clean_noise_type)

    # Initialize model and move to correct device
    model = KAN(width=args.width, grid=args.grid, k=3, seed=1, device=device)
    model.to(device)
    model.fit(dataset, opt=args.opt, steps=args.steps, batch=args.batch_size)

    # Create DataLoaders
    if args.batch_size == -1:
        batch_size = len(train_dataset)
    else:
        batch_size = args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_real_loader = DataLoader(test_real_dataset, batch_size=batch_size, shuffle=False)
    test_real_clean_loader = DataLoader(test_real_clean_dataset, batch_size=batch_size, shuffle=False)
    
    # Evaluate model
    rmse_train = evaluate(train_loader, train_dataset, model, device)
    rmse_test = evaluate(test_loader, test_dataset, model, device)
    rmse_test_real = evaluate(test_real_loader, test_real_dataset, model, device)
    rmse_test_real_clean = evaluate(test_real_clean_loader, test_real_clean_dataset, model, device)

    print(f"RMSE train: {rmse_train:.2e}")
    print(f"RMSE test: {rmse_test:.2e}") # validation set
    print(f"RMSE test_real: {rmse_test_real:.2e}") # test set
    print(f"RMSE test_real_clean: {rmse_test_real_clean:.2e}")

    # Save the RMSE results in a file named based on the arguments
    filename = f"./results/results_dataset{args.train_type}_{args.clean_noise_type}_{args.opt}_steps{args.steps}_batch{args.batch_size}_{args.width}_grid{args.grid}.txt"
    with open(filename, 'w') as f:
        f.write(f"RMSE train: {rmse_train:.2e}\n")
        f.write(f"RMSE test: {rmse_test:.2e}\n")
        f.write(f"RMSE test_real: {rmse_test_real:.2e}\n")