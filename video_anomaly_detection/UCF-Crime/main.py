# from typing import List
import torch
from torch.utils.data import DataLoader
from learner import Learner
from loss import *
from dataset import *
from numpy import argmax
import os
from sklearn import metrics
import matplotlib.pyplot as plt
import argparse

# Declare best_auc as a global variable
best_auc = 0

# Parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch MIL Training')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--w', default=0.0010000000474974513, type=float, help='weight_decay')
    parser.add_argument('--modality', default='TWO', type=str, help='modality')
    parser.add_argument('--input_dim', default=2048, type=int, help='input_dim')
    parser.add_argument('--drop', default=0.6, type=float, help='dropout_rate')
    return parser.parse_args()

# Create datasets and data loaders
def create_datasets_and_loaders(args):
    normal_train_dataset = Normal_dataset(is_train=1, modality=args.modality)
    normal_test_dataset = Normal_dataset(is_train=0, modality=args.modality)
    anomaly_train_dataset = Abnormal_dataset(is_train=1, modality=args.modality)
    anomaly_test_dataset = Abnormal_dataset(is_train=0, modality=args.modality)
    normal_train_loader = DataLoader(normal_train_dataset, batch_size=30, shuffle=True)
    normal_test_loader = DataLoader(normal_test_dataset, batch_size=1, shuffle=True)
    anomaly_train_loader = DataLoader(anomaly_train_dataset, batch_size=30, shuffle=True)
    anomaly_test_loader = DataLoader(anomaly_test_dataset, batch_size=1, shuffle=True)
    num_test_files = anomaly_test_dataset.__len__()
    num_train_files = normal_train_dataset.__len__()
    return normal_train_loader, normal_test_loader, anomaly_train_loader, anomaly_test_loader

# Initialize the model, optimizer, scheduler, and criterion
def initialize_model_optimizer(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Learner(input_dim=args.input_dim, drop_p=args.drop).to(device)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.w)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50])
    criterion = MIL
    return model, optimizer, scheduler, criterion

# Compute the AUC (Area Under the Curve)
def compute_auc(gts, pred):
    fpr, tpr, thresholds = metrics.roc_curve(gts, pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc

# Train the model for one epoch
def train_one_epoch(model, optimizer, scheduler, criterion, normal_train_loader, anomaly_train_loader, device):
    model.train()
    epoch_loss = 0
    for batch_idx, (normal_inputs, anomaly_inputs) in enumerate(zip(normal_train_loader, anomaly_train_loader)):
        inputs = torch.cat([anomaly_inputs, normal_inputs], dim=1)
        batch_size = inputs.shape[0]
        inputs = inputs.view(-1, inputs.size(-1)).to(device)
        outputs = model(inputs)
        itr_loss = criterion(outputs, batch_size)
        epoch_loss += itr_loss.item()
        optimizer.zero_grad()
        itr_loss.backward()
        optimizer.step()
    train_loss = epoch_loss / len(normal_train_loader)
    scheduler.step()
    return train_loss

# Test the model on abnormal data
def test_abnormal(model, anomaly_test_loader, normal_test_loader):
    model.eval()
    global best_auc
    epoch_auc = 0
    with torch.no_grad():
        for i, (data, data2) in enumerate(zip(anomaly_test_loader, normal_test_loader)):
            inputs, gts, frames = data  #Extract graound truths and frame info
            inputs = inputs.view(-1, inputs.size(-1)).to(torch.device('cuda'))
            score = model(inputs)  # Assigning anomaly score to the given videos
            score = score.cpu().detach().numpy()
            score_list = np.zeros(frames[0])
            step = np.round(np.linspace(0, frames[0] // 16, 33)) # Pass asigned score to 
            for j in range(32):
                score_list[int(step[j]) * 16:(int(step[j + 1])) * 16] = score[j]
            gt_list = np.zeros(frames[0])
            for k in range(len(gts) // 2):
                s = gts[k * 2]
                e = min(gts[k * 2 + 1], frames)
                gt_list[s - 1:e] = 1
            inputs2, gts2, frames2 = data2
            inputs2 = inputs2.view(-1, inputs2.size(-1)).to(torch.device('cuda'))
            score2 = model(inputs2)
            score2 = score2.cpu().detach().numpy()
            score_list2 = np.zeros(frames2[0])
            step2 = np.round(np.linspace(0, frames2[0] // 16, 33))
            for kk in range(32):
                score_list2[int(step2[kk]) * 16:(int(step2[kk + 1])) * 16] = score2[kk]
            gt_list2 = np.zeros(frames2[0])
            score_list3 = np.concatenate((score_list, score_list2), axis=0)
            gt_list3 = np.concatenate((gt_list, gt_list2), axis=0)
            itr_auc = compute_auc(gt_list3, score_list3)
            epoch_auc += itr_auc
        test_auc = epoch_auc / len(anomaly_test_loader)
        print('test_auc =', test_auc)
        if best_auc < test_auc:
            print('Saving..')
            state = {
                'net': model.state_dict(),
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            best_auc = test_auc
    print("best auc =", best_auc)

# Main function
def main():
    args = parse_arguments()
    normal_train_loader, normal_test_loader, anomaly_train_loader, anomaly_test_loader = create_datasets_and_loaders(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    model, optimizer, scheduler, criterion = initialize_model_optimizer(args)
    for epoch in range(0, 75):
        train_loss = train_one_epoch(model, optimizer, scheduler, criterion, normal_train_loader, anomaly_train_loader, device)
        print(f'Epoch {epoch}: train_loss = {train_loss:.4f}',)
        test_abnormal(model, anomaly_test_loader, normal_test_loader)

if __name__ == "__main__":
    main()
