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


project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)

from utils.helpers import load_json, recursive_namespace
from utils.dataset import ActionSpotting_v2, ActionSpotting_v2_segments, save_dataset
from utils.evaluate import Checkpoint, save_results
from models_moe.action_selector import SoccerActionSelector


def evaluate(model, testloader, device, eval_mode = 'val'):
    model.eval()
    total_samples = 0
    correct_predictions = 0
    with torch.no_grad():
        for i, (frames, labels, game_name, half, start_time, end_time) in enumerate(testloader):
            frames = frames.to(device)
            labels = labels.to(device)
            class_logits = model(frames)
            preds = torch.argmax(class_logits, dim=-1)  # (B,)
        
            # Compare predictions with targets
            correct_predictions += (preds == labels).sum().item()
            total_samples += targets.size(0)
        
        accuracy = correct_predictions / total_samples
    if eval_mode == 'val':
        wandb.log({"Val Accuracy": accuracy})
    elif eval_mode == 'train':
        wandb.log({"Train Accuracy": accuracy})
        model.train()
        

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
    num_train_samples = min(1000, len(train_dataset))  # Choose the minimum to avoid IndexError
    train_indices = torch.randperm(len(train_dataset))[:num_train_samples]
    subset_train_dataset = Subset(train_dataset, train_indices)
    train_dataloader = DataLoader(subset_train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    num_val_samples = min(300, len(val_dataset))  # Choose the minimum to avoid IndexError
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
    min_lr = 1e-8
    # warmup_scheduler, cosine_scheduler = get_warmup_cosine_scheduler(optimizer, args.warmup_steps, args.num_epochs, min_lr)
    best_ckpt, best_metric = None, 0.0

    for epoch_idx in range(args.num_epochs):
        # if epoch_idx < args.warmup_steps:
        #     warmup_scheduler.step()
        # else:
        #     cosine_scheduler.step(epoch_idx - args.warmup_steps)

        
        training_loss = 0.0
        for batch_idx, (feature_vector, label_vector, game_name, half, start_time, end_time) in enumerate(train_dataloader):
            # print("feature_vector.shape: ", feature_vector.shape)   
            feature_vector = feature_vector[:, ::2, :].to(device)
            label_vector = label_vector.to(device)
            loss, logits = model(feature_vector, label_vector, compute_loss=True)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            training_loss += loss.item()

        print(f"Epoch {epoch_idx}, Loss: {training_loss:.4f}")
        wandb.log({
            'train_loss': training_loss,
            'Epoch': epoch_idx,
            'learning_rate': optimizer.param_groups[0]['lr']
            })
        
        if (epoch_idx) % 5 == 0:
            evaluate(model, train_dataloader, device, 'train')
            evaluate(model, val_dataloader, device, 'val')
        
        # if (epoch_idx + 1) % 15 == 0:
        #     scheduler.step() 

    wandb.finish()
if __name__ == "__main__":
    main(get_args("./configs/moe/actionSpotting_v2_seg_ResNET.json"))
    