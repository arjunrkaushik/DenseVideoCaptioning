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
from utils.dataset import ActionSpotting_v2,  save_dataset
from utils.evaluate import Checkpoint, save_results
from models.actionDetector import ActionDetectionModel2
from models.loss import MatchCriterion


# def compute_IoU_matrix(intervals):
#     """
#     Compute the IoU similarity matrix for a set of temporal intervals.
#     intervals: Tensor of shape (N, 2) where each row contains (start, end)
#     Returns:
#        iou: Tensor of shape (N, N)
#     """
#     # intervals: (N, 2)
#     N = intervals.shape[0]
#     # Expand intervals along two dimensions
#     starts = intervals[:, 0].unsqueeze(1)  # (N, 1)
#     ends   = intervals[:, 1].unsqueeze(1)    # (N, 1)
    
#     # Compute pairwise intersections.
#     inter_start = torch.max(starts, starts.t())  # (N, N)
#     inter_end   = torch.min(ends, ends.t())        # (N, N)
#     intersection = torch.clamp(inter_end - inter_start, min=0)
    
#     # Compute unions.
#     union_start = torch.min(starts, starts.t())
#     union_end   = torch.max(ends, ends.t())
#     union = torch.clamp(union_end - union_start, min=1e-6)
    
#     iou = intersection / union
#     # A tiny constant added to avoid log(0) later.
#     return iou + 1e-6


# def compute_feedback_loss(attn_map, pred_intervals):
#     """
#     Compute the prediction-feedback loss given an attention map and the predicted temporal intervals.
#     attn_map: Tensor of shape (num_queries, num_queries) (already softmax-normalized)
#     pred_intervals: Tensor of shape (num_queries, 2)
    
#     The target relation is defined as the IoU similarity among the predicted intervals.
#     We then use a KL divergence to push the attention relation to resemble this IoU-based relation.
#     """
#     # Compute the target relation (IoU similarity)
#     target_relation = compute_IoU_matrix(pred_intervals)
#     # Normalize target relation by softmax so that it can be compared with the attention map.
#     target_relation = torch.softmax(target_relation, dim=-1)
#     # Note: KLDivLoss in PyTorch expects the input to be log-probabilities.
#     kldiv = nn.KLDivLoss(reduction='batchmean')
#     loss_fb = kldiv(attn_map.log(), target_relation)
#     return loss_fb


# def hungarian_match(pred_logits, pred_boundaries, gt_labels, gt_intervals):
#     """
#     Perform a simple bipartite matching between predictions and ground truth.
#     pred_logits: Tensor of shape (num_queries, num_classes)
#     pred_boundaries: Tensor of shape (num_queries, 2)
#     gt_labels: Tensor of shape (num_targets,) with class indices.
#     gt_intervals: Tensor of shape (num_targets, 2)
    
#     Returns:
#        row_ind: Indices for predictions.
#        col_ind: Indices for matched ground-truth targets.
       
#     The matching cost is a combination of negative log probability (classification cost)
#     and an L1 regression cost.
#     """
#     num_queries = pred_logits.shape[0]
#     num_targets = gt_labels.shape[0]
#     cost_matrix = torch.zeros((num_queries, num_targets)).to(pred_logits.device)
#     # Compute cost per prediction-target pair
#     pred_probs = torch.softmax(pred_logits, dim=-1)
#     for i in range(num_queries):
#         for j in range(num_targets):
#             # Classification cost: negative probability (for the ground-truth class)
#             cls_cost = -pred_probs[i, gt_labels[j]]
#             # Regression cost: L1 distance between predicted and ground truth intervals
#             reg_cost = torch.abs(pred_boundaries[i] - gt_intervals[j]).sum()
#             cost_matrix[i, j] = cls_cost + reg_cost
#     # Use the Hungarian algorithm to compute the minimal cost assignment.
#     row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
#     return row_ind, col_ind


# # ---- Training Step ----

# def train_one_epoch(model, data_loader, optimizer, device, lambda_sa_e=2.0, lambda_sa_d=2.0, lambda_ca_d=2.0):
#     """
#     Runs one epoch of training.
#     Assumes that each batch is a dictionary containing:
#       'frame_embeddings': Tensor (num_frames, batch_size, d_model)
#       'gt_labels': List (length=batch_size) of Tensors (num_targets,) [ground-truth class indices]
#       'gt_intervals': List (length=batch_size) of Tensors (num_targets, 2) [ground-truth (start, end)]
#     The model is expected to output a dictionary containing:
#       'logits': Tensor (batch_size, num_queries, num_classes)
#       'boundaries': Tensor (batch_size, num_queries, 2)
#       Optionally, attention maps from decoder and encoder for feedback:
#         'attn_sa_e', 'attn_sa_d', 'attn_ca_d' – each of shape (batch_size, num_queries, num_queries)
#     """
#     ce_loss_fn = nn.CrossEntropyLoss()
#     l1_loss_fn = nn.L1Loss()
    
#     model.train()
#     total_loss = 0.0
    
#     for batch in data_loader:
#         frame_embeddings = batch['frame_embeddings'].to(device)  
#         # Lists of ground truth (per sample)
#         gt_labels_list = batch['gt_labels']  
#         gt_intervals_list = batch['gt_intervals'] 
        
#         optimizer.zero_grad()
        
#         outputs = model(frame_embeddings)
#         # Primary detection outputs
#         logits = outputs['logits']         # (batch_size, num_queries, num_classes)
#         boundaries = outputs['boundaries']   # (batch_size, num_queries, 2)
        
#         # Optionally provided attention maps for prediction-feedback loss.
#         attn_sa_e = outputs.get('attn_sa_e', None)  # Encoder self-attention
#         attn_sa_d = outputs.get('attn_sa_d', None)  # Decoder self-attention
#         attn_ca_d = outputs.get('attn_ca_d', None)  # Decoder cross-attention
        
#         batch_loss = 0.0
#         batch_size = logits.shape[0]
#         for i in range(batch_size):
#             pred_logits = logits[i]         # (num_queries, num_classes)
#             pred_boundaries = boundaries[i]   # (num_queries, 2)
#             gt_labels = gt_labels_list[i].to(device)       # (num_targets,)
#             gt_intervals = gt_intervals_list[i].to(device)   # (num_targets, 2)
            
#             # Compute bipartite matching between predictions and ground truth.
#             row_ind, col_ind = hungarian_match(pred_logits, pred_boundaries, gt_labels, gt_intervals)
            
#             # Compute set prediction losses.
#             cls_loss = 0.0
#             reg_loss = 0.0
#             num_matches = len(row_ind)
#             for r, c in zip(row_ind, col_ind):
#                 cls_loss += ce_loss_fn(pred_logits[r].unsqueeze(0), gt_labels[c].unsqueeze(0))
#                 reg_loss += l1_loss_fn(pred_boundaries[r], gt_intervals[c])
#             cls_loss = cls_loss / num_matches
#             reg_loss = reg_loss / num_matches
#             loss_detr = cls_loss + reg_loss
            
#             # Compute prediction-feedback losses if the corresponding attention maps are available.
#             feedback_loss = 0.0
#             if attn_ca_d is not None:
#                 # For sample i, assume attn_ca_d[i] has shape (num_queries, num_queries)
#                 feedback_loss += lambda_ca_d * compute_feedback_loss(attn_ca_d[i], pred_boundaries)
#             if attn_sa_d is not None:
#                 feedback_loss += lambda_sa_d * compute_feedback_loss(attn_sa_d[i], pred_boundaries)
#             if attn_sa_e is not None:
#                 feedback_loss += lambda_sa_e * compute_feedback_loss(attn_sa_e[i], pred_boundaries)
                
#             total_sample_loss = loss_detr + feedback_loss
#             batch_loss += total_sample_loss
        
#         batch_loss = batch_loss / batch_size
#         batch_loss.backward()
#         optimizer.step()
#         total_loss += batch_loss.item()
    
#     return total_loss / len(data_loader)


# # ---- Main Training Loop ----

# def main_training_loop(model, train_loader, device, num_epochs=30, lr=1e-4,
#                        lambda_sa_e=2.0, lambda_sa_d=2.0, lambda_ca_d=2.0):
#     optimizer = optim.AdamW(model.parameters(), lr=lr)
#     model.to(device)
    
#     for epoch in range(num_epochs):
#         epoch_loss = train_one_epoch(model, train_loader, optimizer, device,
#                                      lambda_sa_e, lambda_sa_d, lambda_ca_d)
#         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
#         # Optionally, add validation and checkpoint saving here.
    
#     print("Training completed.")

# ---- Example Usage ----

def evaluate(global_step, model, testloader, savedir, device):
    print(f"Testing after {global_step} epochs")

    ckpt = Checkpoint(global_step, bg_class=[0])
    model.eval()
    with torch.no_grad():
        for batch_idx, (feature_vector, label_vector, game_name, half, start_time, end_time) in enumerate(testloader):
            feature_vector = feature_vector.to(device)
            label_vector = label_vector.to(device)

            video_saves = model(feature_vector, label_vector)
            for i in range(1, len(model.block_list)):
                # Log attn map of update block 
                model.block_list[i].f2a_layer.log_attention_map()
                model.block_list[i].a2f_layer.log_attention_map()
            # print(game_name)
            save_results(ckpt, list(game_name), label_vector.detach().tolist(), video_saves, list(start_time), list(end_time))

    model.train()
    ckpt.compute_metrics()

    log_dict = {}
    string = ""
    for k, v in ckpt.metrics.items():
        string += "%s:%.1f, " % (k, v)
        log_dict[f'test-metric/{k}'] = v
    print(string + '\n')
    wandb.log(log_dict)

    fname = "%d.gz" % (global_step) 
    ckpt.save(os.path.join(savedir, fname))

    return ckpt

def get_args(config_file):
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    config = load_json(config_file)
    # args.dataset = SimpleNamespace(**config['dataset'])
        
    # args.device = config['device']
    # args.logdir = config['logdir']
    # args.batch_size = config['batch_size']
    # args.num_epochs = config['num_epochs']
    # args.lr = config['learning_rate']
    # args.wt_decay = config['wt_decay']
    # args.clip_grad_norm = config['clip_grad_norm']
    # args.actionModel = SimpleNamespace(**config['actionModel'])
    args = SimpleNamespace(**config)
    args.actionModel = recursive_namespace(config['actionModel'])
    args.dataset = recursive_namespace(config['dataset'])
    args.matchCriterion = recursive_namespace(config['matchCriterion'])

    run = wandb.init(
        project= 'Action Spotting',
        config=config,  # Use the config dictionary directly
        save_code=False
    )
    return args

def get_lr_lambda(warmup_steps, total_epochs):
    """
    Creates a learning rate lambda function for warmup and decay.
    """
    def lr_lambda(epoch):
        if epoch < warmup_steps:
            return (epoch + 1) / warmup_steps  # Linear warmup
        return 0.95 ** ((epoch - warmup_steps) // 15)  # Step decay every 15 epochs
    return lr_lambda

def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, min_lr):
    """
    Create a learning rate scheduler that applies:
    1. **Linear Warmup** for `warmup_epochs`
    2. **Cosine Annealing with Restarts** after warmup
    """
    def warmup_fn(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs  # Linear warmup
        return 1  # Keep LR unchanged after warmup (until cosine annealing takes over)

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_fn)
    cosine_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=min_lr)

    return warmup_scheduler, cosine_scheduler
def main(args):
    # Run file using 
    # cd /home/csgrad/kaushik3/DenseVideoCaptioning/ActionSpotting
    # python -m main
    device = args.device
    train_dataset = ActionSpotting_v2(feature_path=args.dataset.feature_path, label_path=args.dataset.label_path, split='train', video_length=args.dataset.video_length, overlap=args.dataset.overlap, load_data=args.dataset.load_data)
    val_dataset = ActionSpotting_v2(feature_path=args.dataset.feature_path, label_path=args.dataset.label_path, split='val', video_length=args.dataset.video_length, overlap=args.dataset.overlap, load_data=args.dataset.load_data)
    # train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    # val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    num_train_samples = min(2000, len(train_dataset))  # Choose the minimum to avoid IndexError
    train_indices = torch.randperm(len(train_dataset))[:num_train_samples]
    subset_train_dataset = Subset(train_dataset, train_indices)
    train_dataloader = DataLoader(subset_train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    num_val_samples = min(300, len(val_dataset))  # Choose the minimum to avoid IndexError
    val_indices = torch.randperm(len(val_dataset))[:num_val_samples]
    subset_val_dataset = Subset(val_dataset, val_indices)
    val_dataloader = DataLoader(subset_val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    logdir = args.logdir
    ckptdir = os.path.join(args.logdir, 'ckpts')
    savedir = os.path.join(args.logdir, 'saves')
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(ckptdir, exist_ok=True)
    os.makedirs(savedir, exist_ok=True)
    print('Saving log at', logdir)

    model = ActionDetectionModel2(args = args)
    model.mcriterion = MatchCriterion(args = args, nclasses = args.dataset.num_action_classes, bg_ids=[args.dataset.bg_id])
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
            feature_vector = feature_vector.to(device)
            label_vector = label_vector.to(device)

            optimizer.zero_grad()
            if batch_idx == 0:
                loss, video_saves = model(feature_vector, label_vector, compute_loss=True, visualization_map = True)
            else:
                loss, video_saves = model(feature_vector, label_vector, compute_loss=True)
            loss.backward()

            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

            optimizer.step()
            training_loss += loss.item()

        print(f"Epoch {epoch_idx}, Loss: {training_loss:.4f}")
        wandb.log({
            'train_loss': training_loss,
            'Epoch': epoch_idx,
            'learning_rate': optimizer.param_groups[0]['lr']
            })
        
        if (epoch_idx) % 5 == 0:
            test_ckpt = evaluate(epoch_idx+1, model, val_dataloader, savedir, device)
            if test_ckpt.metrics['F1@0.50'] >= best_metric:
                best_ckpt = test_ckpt
                best_metric = test_ckpt.metrics['F1@0.50']
                print("Best metric updated to", best_metric)

            network_file = ckptdir + '/network.iter-' + str(epoch_idx+1) + '.net'
            model.save_model(network_file)
        
        # if (epoch_idx + 1) % 15 == 0:
        #     scheduler.step()
        
        

    wandb.finish()
if __name__ == "__main__":
    # main(get_args("./configs/actionSpotting_v2_ResNET.json"))
    feature_path = "/data/kaushik3/SoccerData/ResNET_features"
    # feature_path = "/data/kaushik3/SoccerData/Baidu_features"
    label_path = "/data/kaushik3/SoccerData/ActionSpotting/ActionSpotting_v2_segments"
    split = "val"
    video_length = 15
    overlap = 5
    dataset_name = 'ActionSpotting_v2_segments'
    save_dataset(dataset_name, feature_path, label_path, split, video_length, overlap)
    