import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, StepLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Subset
import sys
from scipy.optimize import linear_sum_assignment
from types import SimpleNamespace
from tqdm import tqdm
import wandb
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from collections import defaultdict

project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)

from utils.helpers import load_json, recursive_namespace
from utils.dataset import ActionSpotting_v2, ActionSpotting_v2_segments, save_dataset
from utils.evaluate import Checkpoint, save_results
from models_moe.action_selector import SoccerActionSelector

def compute_dynamic_focal_alpha(trainloader, args, epsilon=1e-6):
    all_labels_list = []
    for i, (frames, labels, game_name, half, start_time, end_time) in enumerate(trainloader):
        # Ensure labels are flattened to 1D for proper counting.
        all_labels_list.append(labels.view(-1))

    # Concatenate all labels along the batch dimension to form one long 1D tensor.
    all_labels = torch.cat(all_labels_list, dim=0)

    # Count frequency for each class, ensuring we have at least num_action_classes bins.
    counts = torch.bincount(all_labels, minlength=args.dataset.num_action_classes).float()

    # Compute inverse frequency and normalize the alpha weights.
    inv_freq = 1.0 / (counts + epsilon)
    dynamic_alpha = inv_freq / inv_freq.sum()

    # Optionally store the computed dynamic alpha into args.
    args.sas.loss.focal_alpha = dynamic_alpha.tolist()
def evaluate(model, testloader, device, eval_mode = 'val'):
    model.eval()
    total_samples = 0
    correct_predictions = 0
    correct_fg = 0
    total_fg_samples = 0
    all_preds = []
    all_labels = []
    num_classes = 18
    class_counts = torch.zeros(num_classes, dtype=torch.long)
    with torch.no_grad():
        for i, (frames, labels, game_name, half, start_time, end_time) in enumerate(testloader):
            frames = frames.to(device)
            labels = labels.to(device)
            label_cnt = labels.view(-1)
            class_counts += torch.bincount(label_cnt.detach().cpu(), minlength=num_classes)
            class_logits = model(frames)
            all_preds.append(class_logits.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
            preds = torch.argmax(class_logits, dim=-1)  # (B,)
            # print(preds.shape)
            # Compare predictions with targets
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(1)*labels.size(0)

            # Calculate accuracy for foreground frames only
            labels = labels.view(-1)
            preds = preds.view(-1)
            fg_mask = (labels > 0)  # True for foreground frames
            total_fg = fg_mask.sum().item()

            # Handle the case when there are no non-background frames.
            if total_fg > 0:
                correct_fg += (preds[fg_mask] == labels[fg_mask]).sum().item()
                total_fg_samples += total_fg
        
        accuracy = 100* correct_predictions / total_samples
        foreground_accuracy = 100*correct_fg / total_fg_samples

    all_labels = np.concatenate(all_labels).reshape(-1)
    mlb = MultiLabelBinarizer(classes=np.arange(18))  # Define classes from 0 to 17
    all_labels_one_hot = mlb.fit_transform(all_labels.reshape(-1, 1)) 
    all_preds = np.concatenate(all_preds).reshape(-1, 18)
    AP = []
    for i in range(0, 18):
        AP.append(average_precision_score(all_labels_one_hot[:, i], all_preds[:, i]))
    mAP = np.mean(AP)
    for class_id, count in enumerate(class_counts.tolist()):
        print(f"Class {class_id}: {count} samples")
    if eval_mode == 'val':
        wandb.log({
            "Val Accuracy With BG": accuracy,
            "Val Accuracy Without BG": foreground_accuracy,
            "Val mAP": mAP})
        # print("Total val samples: ", total_samples)
        # print("Total val fg samples: ", total_fg_samples)
    elif eval_mode == 'train':
        wandb.log({
            "Train Accuracy With BG": accuracy,
            "Train Accuracy Without BG": foreground_accuracy,
            "Train mAP": mAP})
        
        # print("Total train samples: ", total_samples)
        # print("Total train fg samples: ", total_fg_samples)
        

def get_args(config_file):
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    config = load_json(config_file)
    args = SimpleNamespace(**config)
    args.sas = recursive_namespace(config['sas'])
    args.dataset = recursive_namespace(config['dataset'])

    run = wandb.init(
        project= 'Action Spotting',
        config=config,  # Use the config dictionary directly
        save_code=False
    )
    return args

def main(args):
    # Run file using 
    # cd /home/csgrad/kaushik3/DenseVideoCaptioning/ActionSpotting
    # python -m main
    device = args.device
    train_dataset = ActionSpotting_v2_segments(feature_path=args.dataset.feature_path, label_path=args.dataset.label_path, split='train', video_length=args.dataset.video_length, overlap=args.dataset.overlap, load_data=args.dataset.load_data)
    val_dataset = ActionSpotting_v2_segments(feature_path=args.dataset.feature_path, label_path=args.dataset.label_path, split='val', video_length=args.dataset.video_length, overlap=args.dataset.overlap, load_data=args.dataset.load_data)
    # train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    # val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    num_train_samples = min(8000, len(train_dataset))  # Choose the minimum to avoid IndexError
    train_indices = torch.randperm(len(train_dataset))[:num_train_samples]
    subset_train_dataset = Subset(train_dataset, train_indices)
    train_dataloader = DataLoader(subset_train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    compute_dynamic_focal_alpha(train_dataloader, args)

    num_val_samples = min(800, len(val_dataset))  # Choose the minimum to avoid IndexError
    val_indices = torch.randperm(len(val_dataset))[:num_val_samples]
    subset_val_dataset = Subset(val_dataset, val_indices)
    val_dataloader = DataLoader(subset_val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    logdir = args.logdir
    ckptdir = os.path.join(args.logdir, 'ckpts')
    # savedir = os.path.join(args.logdir, 'saves')
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(ckptdir, exist_ok=True)
    # os.makedirs(savedir, exist_ok=True)
    print('Saving log at', logdir)

    model = SoccerActionSelector(args = args)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay = args.wt_decay)
    # scheduler = LambdaLR(optimizer, lr_lambda=get_lr_lambda(args.warmup_steps, args.num_epochs))
    scheduler = StepLR(optimizer, step_size=40, gamma=0.5)
    min_lr = 1e-8
    # warmup_scheduler, cosine_scheduler = get_warmup_cosine_scheduler(optimizer, args.warmup_steps, args.num_epochs, min_lr)
    best_ckpt, best_metric = None, 0.0

    for epoch_idx in range(args.num_epochs):
        # if epoch_idx < args.warmup_steps:
        #     warmup_scheduler.step()
        # else:
        #     cosine_scheduler.step(epoch_idx - args.warmup_steps)

        
        training_loss = 0.0
        model.train()
        epoch_loss_dict = defaultdict(float)
        for batch_idx, (feature_vector, label_vector, game_name, half, start_time, end_time) in enumerate(train_dataloader):
            # print("feature_vector.shape: ", feature_vector.shape)   
            feature_vector = feature_vector.to(device)
            label_vector = label_vector.to(device)
            if batch_idx == 5: 
                loss, loss_dict, logits = model(feature_vector, label_vector, compute_loss=True, plot = True)
            else:
                loss, loss_dict, logits = model(feature_vector, label_vector, compute_loss=True)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            training_loss += loss.item()
            for key, value in loss_dict.items():
                epoch_loss_dict[key] += value.item()
        scheduler.step()
        print(f"Epoch {epoch_idx}, Loss: {training_loss:.4f}")
        wandb.log({
            'train_loss': training_loss,
            'Epoch': epoch_idx,
            'learning_rate': optimizer.param_groups[0]['lr']
            })
        for key, value in epoch_loss_dict.items():
            wandb.log({'Train ' + key: value})
        
        if (epoch_idx) % 5 == 0:
            evaluate(model, train_dataloader, device, 'train')
            evaluate(model, val_dataloader, device, 'val')
        
        # if (epoch_idx + 1) % 15 == 0:
        #     scheduler.step() 

    wandb.finish()
if __name__ == "__main__":
    main(get_args("./configs/moe/actionSpotting_v2_seg_ResNET.json"))
    