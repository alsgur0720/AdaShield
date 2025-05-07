import os, random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from network import DRS, differential_detection
from load_data import *
from loss import *
import torch.nn as nn
from experiments import exp_DRS as exp
import wandb
import cma
import pickle
import networkx as nx
import time
wandb.init(project="federated-learning", name="FL_DRS", config={"epochs": exp.train_epoch})

def compute_second_derivative(data_segment):
    first_derivative = torch.diff(data_segment, n=1, dim=1)
    second_derivative = torch.diff(first_derivative, n=1, dim=1)
    return second_derivative


def compute_first_derivative(data_segment):
    first_derivative = torch.diff(data_segment, n=1, dim=1)
    return first_derivative


def loss_function(beta, original_loss, local_cov, global_cov):
    """
    Compute the loss for a given beta.
    
    Args:
        beta (float): Weighting factor for the Frobenius norm.
        original_loss (float): Original loss value (e.g., cross-entropy loss).
        local_cov (torch.Tensor): Local covariance matrix (C, C).
        global_cov (torch.Tensor): Global covariance matrix (C, C).
    
    Returns:
        float: The computed loss.
    """
    # Convert beta to a tensor (since CMA-ES passes numpy arrays)
    beta = torch.tensor(beta, dtype=torch.float32)
    
    # Compute Frobenius norm
    frobenius_norm = torch.norm(local_cov - global_cov, p='fro').item()
    
    beta_value = beta.item()
    # Compute total loss
    total_loss = (1-beta_value) * original_loss + beta_value * frobenius_norm
    
    return float(total_loss)


def objective_function(beta_array, original_loss, local_cov, global_cov):
        # CMA-ES provides beta as an array, so unpack it
    beta = beta_array[0]
    return loss_function(beta, original_loss, local_cov, global_cov)

def optimize_beta(original_loss, local_cov, global_cov):
    """
    Optimize the beta parameter using CMA-ES.
    
    Args:
        original_loss (float): Original loss value.
        local_cov (torch.Tensor): Local covariance matrix (C, C).
        global_cov (torch.Tensor): Global covariance matrix (C, C).
    
    Returns:
        float: Optimized beta value.
    """
    # Define the objective function for CMA-ES
    objective_function = lambda beta_array: loss_function(beta_array[0], original_loss, local_cov, global_cov)
    
    # Initialize CMA-ES
    initial_beta = np.array([0.1], dtype=np.float64)  # Initial guess for beta
    sigma = 0.5  # Standard deviation for search
    bounds = [0.0, 10.0]  # Limit beta to a reasonable range
    print(f"initial_beta: {initial_beta}, type: {type(initial_beta)}, shape: {initial_beta.shape}")
    print(f"bounds: {bounds}")

    # Run CMA-ES
    es = cma.CMAEvolutionStrategy(
        initial_beta, sigma, {'verbose': -9, 'bounds': bounds},
        
    )
    optimized_beta = es.optimize(objective_function).result.xbest[0]
    optimized_beta = max(bounds[0], min(bounds[1], optimized_beta))
    return optimized_beta

class CustomDataset(Dataset):
    def __init__(self, client_id, train=True):
        self.client_id = client_id
        self.train = train
        self.adver_client = []
        
        self.load_data()
        
    def __len__(self):
        return len(self.data)

    def load_data(self):
        base_dir = 'data_3' if self.client_id in self.adver_client else 'data_2'
        client_data_dir = f'./{base_dir}/Client{self.client_id}/raw_data'
        # process_data_dir = f'./{base_dir}/Client{self.client_id}/process_data'
        if self.train :
            process_data_dir = f'./{base_dir}/Client{self.client_id}/process_data'
        else:
            process_data_dir = f'./{base_dir}/Client{self.client_id}/process_data_test'

        print('self.client_id:', self.client_id)

        # 조건에 따른 read_data 호출
        if self.client_id == 4 or self.client_id == 8:
            dataset = read_data(client_data_dir, process_data_dir, exp.image_size, exp.n_slice, is_drs=exp.is_drs, is_acdc=True)
        
        elif self.client_id == 1 or self.client_id == 2 or self.client_id == 3 or self.client_id == 5 or self.client_id == 6:
            dataset = read_data(client_data_dir, process_data_dir, exp.image_size, exp.n_slice, is_drs=exp.is_drs, is_acdc=False, is_adver = False)
        else: 
            dataset = read_data(client_data_dir, process_data_dir, exp.image_size, exp.n_slice, is_drs=exp.is_drs, is_acdc=exp.is_acdc)
        
    
        images_train_3D, gts_train_3D, images_validation_3D, gts_validation_3D = adjust_data_3D(
            dataset['images_train'], dataset['gts_train'],
            dataset['images_validation'], dataset['gts_validation'],
            exp.image_size, exp.n_slice
        )
        
        if self.train:
            self.data, self.labels = data_augmentation_3D(images_train_3D, gts_train_3D, flip=True, gamma=True)
            
        else:
            self.data, self.labels = images_validation_3D, gts_validation_3D

            
    def __getitem__(self, idx): 
        image, label = self.data[idx], self.labels[idx]
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)




def malicious_client_purification(model, images):
    model.eval()
    output_seg, cov, b_seg, purified_image, b_rec = model(images)
    for i in range(0,2):
        _, _, _, purified_image, _ = model(purified_image)
    
    return purified_image


def train_epoch(model, epoch, loader, criterion, criterion_rec, criterion_inde, optimizer, device, global_cov, is_malicious):
    model.train()
    epoch_loss = 0.0
    local_cov = None
    local_mean = None
    total_samples = 0
    optimized_beta = 0.0
    loss_cov = 0.0
    avg_len = 0
    epoch_dice = 0.0
    epoch_segt = 0.0
    epoch_rec = 0.0
    epoch_inde = 0.0
    epoch_cov = 0.0
    print(f"Training epoch {epoch}")
    
    loss_save = []
    
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        images = images.permute(0,3,1,2)
        
        
        if is_malicious:
            purified_image = malicious_client_purification(model, images)
            images = purified_image
        
        optimizer.zero_grad()

        output_seg, cov, b_seg, output_rec, b_rec = model(images)
        
        loss_rec = criterion_rec(output_rec, images)
        
        loss_segt, accuracy_segt, _, _, _, _, _, _ = criterion(output_seg, labels, 4)
     
        loss_inde = criterion_inde(b_seg, b_rec)
        
        loss = loss_segt + loss_inde + loss_rec
        
        if global_cov is not None:
            
            optimized_beta = optimize_beta(loss, cov, global_cov)
            loss_cov = torch.norm(cov - global_cov, p='fro').item()
            loss = (1 - optimized_beta) * loss + optimized_beta * loss_cov
            
            ## ablation
            # loss = (1 - 0.7) * loss + 0.7 * loss_cov

        batch_size, C, H, W = b_seg.size()
        batch_mean = b_seg.mean(dim=(0, 2, 3))
        spatial_samples = batch_size * H * W
        if local_cov is None:
            # First batch initialization
            local_cov = cov * (spatial_samples - 1)  # Weighted batch covariance
            local_mean = batch_mean * spatial_samples
            total_samples = spatial_samples
        else:
            # Update local covariance and mean
            delta = batch_mean - (local_mean / total_samples)
            local_cov += cov * (spatial_samples - 1)  # Add weighted batch covariance
            local_cov += spatial_samples * total_samples * torch.outer(delta, delta) / (spatial_samples + total_samples)
            local_mean += batch_mean * spatial_samples
            total_samples += spatial_samples
            
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_segt += loss_segt.item()
        epoch_rec += loss_rec.item()
        epoch_inde += loss_inde.item()
        epoch_dice += accuracy_segt
        epoch_cov += loss_cov
        
# Final local covariance normalization
    local_cov /= (total_samples - 1)
    local_mean /= total_samples
    avg_dice = epoch_dice / len(loader)
    avg_loss = epoch_loss / len(loader)
    avg_segt = epoch_segt / len(loader)
    avg_rec = epoch_rec / len(loader)
    avg_inde = epoch_inde / len(loader)
    avg_cov = epoch_cov / len(loader)
    print('  loss:\t\t{:.6f}'.format(avg_loss))
    print('  accuracy:\t\t{:.6f}'.format(avg_dice))
    
    wandb.log({f"Client_{client_id}_Train_cov_seg_Loss": avg_loss, 
               "epoch": epoch, f"beta_{client_id}_": optimized_beta, 
               f"loss_{client_id}_cov": avg_cov,
               f"loss_{client_id}_seg": avg_segt,
               f"loss_{client_id}_rec": avg_rec,
               f"loss_{client_id}_inde": avg_inde})    
    
    return avg_loss, local_cov, local_mean, avg_segt

def validate_epoch(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    avg_len = 0
    avg_dice = 0.0
    avg_accuracy = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            images = images.permute(0,3,1,2)
            output_seg, cov, b_seg, output_rec, b_rec = model(images)
            
            loss, accuracy_segt, _, _, _, _, _, _ = criterion(output_seg, labels, 4)
            val_loss += loss.item()
            avg_accuracy += accuracy_segt
    # avg_val_loss = val_loss / avg_len
    # avg_val_dice = avg_dice / avg_len
    avg_val_loss = val_loss / len(loader)
    avg_accuracy = avg_accuracy / len(loader)
    wandb.log({"Global_Validation_Loss": avg_val_loss, "epoch": epoch, "Global_accuracy": accuracy_segt})
    return avg_val_loss


def fedavg_aggregation(client_models):
    n_clients = len(client_models)
    global_state_dict = client_models[0].state_dict()
    for key in global_state_dict.keys():
        param_list = torch.stack([client_models[i].state_dict()[key].float() for i in range(n_clients)])
        global_state_dict[key] = torch.mean(param_list, dim=0)
        
    return global_state_dict


def attack_aware_global_aggregation(client_models, global_model):

    n_clients = len(client_models)
    global_state_dict = global_model.state_dict()
    
    # Initialize a dictionary to store the sum of gradients for each parameter
    gradient_sums = {key: torch.zeros_like(param) for key, param in global_state_dict.items()}
    
    for client_model in client_models:
        client_state_dict = client_model.state_dict()
        
        for key in global_state_dict.keys():
            # Compute the gradient (global parameter - client parameter)
            gradient = client_state_dict[key] - global_state_dict[key]
            
            # Accumulate the gradient
            gradient_sums[key] += gradient
    
    # Update the global model using the average gradient
    for key in global_state_dict.keys():
        global_state_dict[key] += 0.1 * (gradient_sums[key] / n_clients)
    
    # Load the updated state dictionary into the global model
    global_model.load_state_dict(global_state_dict)
    
    return global_model


def PearsonCorrelationLoss(A,B):

    mean_A = torch.mean(A, dim=0)
    mean_B = torch.mean(B, dim=0)
    A_centered = A - mean_A
    B_centered = B - mean_B
    covariance = torch.mean(A_centered * B_centered, dim=0)
    std_A = torch.std(A, dim=0)
    std_B = torch.std(B, dim=0)
    correlation = covariance / (std_A * std_B + 1e-8)
    loss = torch.mean(correlation**2)
    return loss

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    global_model = DRS().to(device)
    detection = differential_detection().to(device)
    detection.load_state_dict(torch.load(exp.detection_model_path))
    criterion = torch_loss_accuracy_3D
    criterion_rec = nn.MSELoss()
    criterion_cov = PearsonCorrelationLoss
    num_epochs = exp.train_epoch
    best_val_loss = float('inf')

    client_loaders = []
    for client_id in range(1, 9):  # Client 1 to Client 4
        train_dataset = CustomDataset(client_id=client_id, train=True)
        train_loader = DataLoader(train_dataset, batch_size=exp.train_batch_size, shuffle=True, num_workers=2)
        client_loaders.append(train_loader)

    val_dataset = CustomDataset(client_id=1, train=False)
    val_loader = DataLoader(val_dataset, batch_size=exp.validation_batch_size, shuffle=False, num_workers=2)

    global_cov = None
    global_mean = None
    global_cov_prev = None
    
    num_round = 300
    
    total_samples = 0
    loss_save_round = []
    loss_path = './loss_save'
    excluded_client_ids = []
    
    for round_idx in range(num_round):
        print(f"Round [{round_idx+1}/{num_round}] - Federated Learning with Averaging Aggregation")
        start_time_round = time.time()        
        # Federated Training Step
        client_models_temp = []
        client_models = []
        client_loss_save = []
        local_covs = []  
        local_means = []  
        sample_counts = []
        client_ids = []
        for client_id, train_loader in enumerate(client_loaders, start=1):
            print(f"Training on client {client_id}")
            model = DRS().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=exp.learning_rate)
            
            model.load_state_dict(global_model.state_dict())
            
            for epoch in range(num_epochs):  # Local training loop
                if round_idx >= 1 :
                    if client_id in excluded_client_ids:
                        train_loss, local_cov, local_mean, loss_save = train_epoch(model, epoch, train_loader, 
                                                                    criterion, criterion_rec, criterion_cov, optimizer, 
                                                                    device, global_cov_prev, is_malicious=True)
                    else:
                        train_loss, local_cov, local_mean, loss_save = train_epoch(model, epoch, train_loader, 
                                                                    criterion, criterion_rec, criterion_cov, optimizer, 
                                                                    device, global_cov_prev, is_malicious=False)
                else:
                    train_loss, local_cov, local_mean, loss_save = train_epoch(model, epoch, train_loader, 
                                                                criterion, criterion_rec, criterion_cov, optimizer, 
                                                                device, None, is_malicious=False)
                client_loss_save.append(loss_save)    
                
            print(f"Client {client_id} - Train Loss: {train_loss:.4f}")
            
            client_checkpoint_dir = os.path.join(exp.checkpoint_model_dir, f"client_{client_id}")
            os.makedirs(client_checkpoint_dir, exist_ok=True)
            client_checkpoint_path = os.path.join(client_checkpoint_dir, f"model_epoch_{round_idx+1}.pth")
            torch.save(model.state_dict(), client_checkpoint_path)
            print(f"Client {client_id} model for epoch {round_idx+1} saved at {client_checkpoint_path}")
                        
            client_models_temp.append(model)
            
            client_sample_count = sum(len(batch[0]) for batch in train_loader)

            local_covs.append(local_cov)
            local_means.append(local_mean)
            sample_counts.append(client_sample_count)
            client_ids.append(client_id)
                        
            # if global_cov is None:
            #     global_cov = local_cov * (client_sample_count - 1)  # Weighted covariance
            #     global_mean = local_mean * client_sample_count
            #     total_samples = client_sample_count
            # else:
            #     # Update global stats
            #     global_mean_prev = global_mean / total_samples  # Previous global mean
            #     delta = local_mean - global_mean_prev
            #     global_cov += local_cov * (client_sample_count - 1)  # Add weighted covariance
            #     global_cov += client_sample_count * total_samples * torch.outer(delta, delta) / (client_sample_count + total_samples)
            #     global_mean += local_mean * client_sample_count
            #     total_samples += client_sample_count
        
        
        
        client_loss_save = torch.tensor(client_loss_save, dtype=torch.float32)
        loss_save_round.append(client_loss_save)
        
        range_indices = [(0, 100), (100, 200), (200, 300), (300, 400), (400, 500), (500, 600), (600, 700), (700, 800)]
        
        gradient_tensors = []
        for idx, data in enumerate(loss_save_round, start=1):
            first_derivative_tensors = []
            for start_idx, end_idx in range_indices:
                data_segment = data[start_idx:end_idx]
                first_derivative = compute_first_derivative(data_segment)
                first_derivative_tensors.append(first_derivative)
            
            gradient_tensor = torch.cat(first_derivative_tensors, dim=1)
            gradient_tensors.append(gradient_tensor)
        gradient_tensors = torch.cat(gradient_tensors, dim=0)

        
        curvature_tensors = []
        for idx, data in enumerate(loss_save_round, start=1):
            second_derivative_tensors = []
            for start_idx, end_idx in range_indices:
                data_segment = data[start_idx:end_idx]
                second_derivative = compute_second_derivative(data_segment)
                second_derivative_tensors.append(second_derivative)
            
            curvature_tensor = torch.cat(second_derivative_tensors, dim=1)
            curvature_tensors.append(curvature_tensor)
        
        curvature_tensors = torch.cat(curvature_tensors, dim=0)
        
        final_loss = torch.cat((gradient_tensor, curvature_tensor), dim=1)
        
        detection.eval()
        
        malicious_status = detection(final_loss)
                 
        
        for i in range(1,9):
            if malicious_status[i] < 0.1:
                excluded_client_ids.append(i)
            
        
        os.makedirs(loss_path, exist_ok=True)
        os.makedirs(f'{loss_path}/round_{round_idx+1}', exist_ok=True)
        with open(f'{loss_path}/round_{round_idx+1}/loss_save_round.pkl', 'wb') as f:
            pickle.dump(np.array(loss_save_round), f)
        # np.save(f'{loss_path}/round_{round_idx+1}/loss_save_round.npy', loss_save_round)
        print(f"Round {round_idx+1} loss saved at {loss_path}/round_{round_idx+1}/loss_save_round.npy")
              
        
        for excluded_client_id in excluded_client_ids:
            if excluded_client_id in client_ids:
                exclude_idx = client_ids.index(excluded_client_id)  
                local_covs.pop(exclude_idx)  
                local_means.pop(exclude_idx)  
                sample_counts.pop(exclude_idx)
                client_models_temp.pop(exclude_idx)  
                client_ids.pop(exclude_idx)  
           
        if len(local_covs) > 0:  
            total_samples = sum(sample_counts)
            global_mean = sum(local_mean * count for local_mean, count in zip(local_means, sample_counts)) / total_samples
            global_cov = torch.zeros_like(local_covs[0])
            
            for local_cov, local_mean, count in zip(local_covs, local_means, sample_counts):
                delta = local_mean - global_mean
                global_cov += local_cov * (count - 1)  # Weighted covariance
                global_cov += count * torch.outer(delta, delta) / total_samples  # Mean adjustment
        
        
        
        if exp.aggregation_method == "AdaShiled-FL":
            aggregated_state_dict = attack_aware_global_aggregation(client_models_temp)
        elif exp.aggregation_method == "fed_avg":
            aggregated_state_dict = fedavg_aggregation(client_models_temp)
        else:
            raise ValueError("Unknown aggregation method specified")

        # Update global model with aggregated parameters
        global_model.load_state_dict(aggregated_state_dict)

        # Validation Step (Global Model)
        val_loss = validate_epoch(global_model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}")

        # Model Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(exp.checkpoint_model_dir, "best_model.pth")
            torch.save(global_model.state_dict(), checkpoint_path)
            print(f"New best model saved at {checkpoint_path} with Validation Loss: {val_loss:.4f}")