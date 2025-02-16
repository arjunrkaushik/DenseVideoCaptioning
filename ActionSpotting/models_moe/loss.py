import torch
import torch.nn as nn
import torch.nn.functional as F

class MoeLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.alpha = args.sas.loss.alpha  # Weight for diversity loss
        self.beta = args.sas.loss.beta # Weight for contrastive loss
        self.temperature = args.sas.loss.temperature
        self.label_smoothing = args.sas.loss.label_smoothing
        self.num_experts = args.sas.num_experts
        
        # Base classification loss with label smoothing
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        
    def diversity_loss(self, expert_probs):
        """Encourages expert specialization through covariance regularization"""
        # expert_probs shape: (batch, frames, num_experts)
        batch_size, num_frames, _ = expert_probs.shape
        
        # Calculate mean expert probabilities across batch and frames
        mean_probs = expert_probs.mean(dim=[0,1])  # [num_experts]
        
        # Compute covariance matrix
        cov = torch.outer(mean_probs, mean_probs)  # [num_experts, num_experts]
        
        # Penalize off-diagonal correlations (experts being similar)
        diversity_loss = (cov.sum() - cov.diag().sum()) / (self.num_experts * (self.num_experts - 1))
        
        return diversity_loss
    
    def contrastive_loss(self, frame_embs, proj_text, frame_targets):
        """
        Computes the contrastive loss between each frame embedding and projected text embeddings.
        
        For each frame (of shape [D]), the cosine similarities are computed against every
        action label's projected text embedding (of shape [num_actions, D]). Using the ground truth label
        for that frame as the target, a cross entropy loss is applied.
        
        Args:
            frame_embs (Tensor): Frame embeddings of shape (B, 15, D)
            proj_text (Tensor): Projected text embeddings of action labels, shape (num_actions, D)
            frame_targets (Tensor): Ground truth action labels per frame, shape (B, 15),
                                    with integer entries in [0, num_actions-1]
        
        Returns:
            Tensor: Scalar contrastive loss.
        """
        B, T, D = frame_embs.shape
        # Reshape frames to (B*T, D)
        frames = frame_embs.view(B * T, D)
        
        # Normalize both sets of embeddings to unit length (for cosine similarity)
        frames = F.normalize(frames, p=2, dim=-1)
        proj_text = F.normalize(proj_text, p=2, dim=-1)  # (num_actions, D)
        
        # Compute similarity matrix: (B*T, num_actions)
        sim_matrix = torch.matmul(frames, proj_text.t()) / self.temperature
        
        # Flatten frame_targets to shape (B*T,)
        targets = frame_targets.view(B * T)
        
        # Use cross entropy loss on the similarity scores where the correct text embedding is the target
        loss_contrast = F.cross_entropy(sim_matrix, targets)
        return loss_contrast

    def forward(self, pred_logits, frame_targets, expert_probs, frame_embs, proj_text):
        """
        Args:
            pred_logits (Tensor): Aggregated video-level classification logits from MoE, shape (B, num_actions)
            video_targets (Tensor): Ground truth video-level labels, shape (B,)
            expert_probs (Tensor): Expert routing probabilities, shape (B, 15, num_experts)
            frame_embs (Tensor): Frame embeddings (before MoE aggregation), shape (B, 15, D)
            proj_text (Tensor): Projected text embeddings for action labels, shape (num_actions, D)
            frame_targets (Tensor): Ground truth action label for each frame, shape (B, 15)
        
        Returns:
            Tensor: Combined loss.
        """
        # Classification loss for aggregated predictions
        B, T, D = pred_logits.shape
        # Reshape frames to (B*T, D)
        pred_logits = pred_logits.view(B * T, D)
        loss_cls = self.ce_loss(pred_logits, frame_targets.view(-1))
        
        # Diversity loss for encouraging expert specialization
        loss_div = self.diversity_loss(expert_probs)
        
        # Contrastive loss comparing each frame's embedding to the projected text embeddings
        # loss_contrast = self.contrastive_loss(frame_embs, proj_text, frame_targets)
        
        total_loss = loss_cls + self.alpha * loss_div 
        return total_loss
