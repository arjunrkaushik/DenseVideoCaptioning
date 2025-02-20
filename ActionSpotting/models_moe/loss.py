import torch
import torch.nn as nn
import torch.nn.functional as F

class MoeLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.alpha = args.sas.loss.alpha  # Weight for diversity loss
        self.beta = args.sas.loss.beta  # Weight for contrastive loss
        self.gamma = args.sas.loss.gamma  # Weight for expert focal loss
        self.delta = args.sas.loss.delta  # Weight for contrastive loss on text embeddings
        self.temperature = args.sas.loss.temperature
        self.label_smoothing = args.sas.loss.label_smoothing
        self.num_experts = args.sas.num_experts
        self.focal_alpha = torch.tensor(args.sas.loss.focal_alpha).to(args.device) 
        
        # Focal loss parameters
        self.focal_gamma = args.sas.loss.focal_gamma  # e.g., 2.0

    def compute_batch_balanced_alpha(self, targets, num_classes, epsilon=1e-6):
        """
        Computes dynamic α weights for focal loss based on the batch's class distribution.
        
        Args:
            targets (Tensor): Ground-truth labels of shape (B*T,) or (B, T).
            num_classes (int): Total number of classes.
            epsilon (float): Small constant to avoid division by zero.
        
        Returns:
            Tensor: Per-class weights α of shape (num_classes,).
        """
        # Flatten targets if needed
        targets_flat = targets.view(-1)
        
        # Count occurrences of each class in the batch
        counts = torch.bincount(targets_flat, minlength=num_classes).float()
        
        # Compute inverse frequency for each class
        inv_freq = 1.0 / (counts + epsilon)
        
        # Normalize so that weights sum to 1
        alpha = inv_freq / inv_freq.sum()
        
        return alpha

    def focal_loss_fn(self, inputs, targets, alpha=None):
        """
        Computes the focal loss with dynamic or static α balancing.
        
        Args:
            inputs (Tensor): Logits of shape (N, C), where C is the number of classes.
            targets (Tensor): Ground-truth labels of shape (N,) with integer class indices.
            alpha (Tensor or None): Per-class weights α of shape (C,) or scalar weight.
        
        Returns:
            Tensor: The scalar focal loss.
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        
        # Compute p_t (the probability of the true class)
        pt = torch.exp(-ce_loss)
        
        # Apply dynamic or static α balancing
        if alpha is not None:
            alpha_weight = alpha.gather(0, targets)  # Gather α for each target class
        else:
            alpha_weight = 1.0  # Default scalar weight
        
        focal_loss = alpha_weight * ((1 - pt) ** self.focal_gamma) * ce_loss
        return focal_loss.mean()

    def diversity_loss(self, expert_probs):
        """Encourages expert specialization through covariance regularization."""
        mean_probs = expert_probs.mean(dim=[0, 1])  # Shape: [num_experts]
        
        # Compute covariance matrix
        cov = torch.outer(mean_probs, mean_probs)  # Shape: [num_experts, num_experts]
        
        # Penalize off-diagonal correlations (experts being similar)
        diversity_loss = (cov.sum() - cov.diag().sum()) / (self.num_experts * (self.num_experts - 1))
        
        return diversity_loss

    def contrastive_learning(self, frame_embs, proj_text, frame_targets):
        """Computes contrastive loss using NT-Xent (InfoNCE)."""
        B, T, D = frame_embs.shape
        
        frame_embs = F.normalize(frame_embs, p=2, dim=-1)
        proj_text = F.normalize(proj_text, p=2, dim=-1)  # Shape: [num_actions, D]
        
        sim_matrix = torch.matmul(frame_embs, proj_text.transpose(1, 2)) / self.temperature  # Shape: [B*T, num_actions]
        
        sim_matrix = sim_matrix.view(B * T, -1)  # Reshape for easier loss computation
        
        targets_one_hot = F.one_hot(frame_targets.view(-1), num_classes=sim_matrix.shape[-1]).float()
        
        log_probs = F.log_softmax(sim_matrix, dim=-1)
        
        loss_contrast = -torch.sum(targets_one_hot * log_probs) / targets_one_hot.sum()
        
        return loss_contrast
    
    def contrastive_text_loss(self, proj_text):
        """
        Computes contrastive loss for projected text embeddings.
        - Since text projections are identical per class, we only push other classes away.

        Args:
            proj_text (Tensor): Projected text embeddings of shape (B, 18, 2048).

        Returns:
            Tensor: Contrastive loss for text embeddings.
        """
        B, num_classes, D = proj_text.shape  # (B, 18, 2048)
        
        # Normalize embeddings
        proj_text = F.normalize(proj_text, p=2, dim=-1)  # (B, 18, 2048)

        # Reshape to treat the same classes across batches as a single entity
        proj_text_flat = proj_text.view(B * num_classes, D)  # (B * 18, 2048)
        
        # Compute cosine similarity matrix
        sim_matrix = torch.matmul(proj_text_flat, proj_text_flat.T)  # (B*18, B*18)
        
        # Apply temperature scaling
        sim_matrix /= self.temperature

        # Create mask to ignore same-class comparisons (we do NOT pull positives together)
        text_labels = torch.arange(num_classes, device=proj_text.device).repeat(B)  # (B * 18,)
        negative_mask = (text_labels.unsqueeze(0) != text_labels.unsqueeze(1)).float()  # (B*18, B*18)

        # Compute loss: Push negatives away using softmax over all classes
        exp_sim = torch.exp(sim_matrix) * negative_mask  # Zero out same-class values
        exp_sim_sum = exp_sim.sum(dim=-1, keepdim=True)  # Sum over negative classes

        # Compute final loss by applying log softmax over negatives
        loss = -torch.log(1 - exp_sim / (exp_sim_sum + 1e-6)).mean()  # Avoid log(0) with epsilon

        return loss

    def forward(self, pred_logits, frame_targets, expert_probs, frame_embs, proj_text, transformed_frames, transformed_text):
        """
        Combines BBFL-based focal loss with diversity and contrastive losses.
        
        Args:
            pred_logits (Tensor): Classification logits of shape [B*T, num_actions].
            frame_targets (Tensor): Ground-truth labels of shape [B*T].
            expert_probs (Tensor): Expert routing probabilities of shape [B*T].
            frame_embs (Tensor): Frame embeddings of shape [B*T].
            proj_text (Tensor): Projected text embeddings of shape [num_actions].
        
        Returns:
            Tensor: Combined loss value.
        """
        
        B, T, num_classes = pred_logits.shape
        
        # Flatten logits and targets for cross-entropy-based losses
        # print(f"pred_logits.shape: {pred_logits.shape}")
        pred_logits_flat = pred_logits.view(B * T, num_classes)
        expert_probs_flat = expert_probs.view(B * T, -1)
        frame_targets_flat = frame_targets.view(-1)
        
        # Compute dynamic α for BBFL
        dynamic_alpha = self.compute_batch_balanced_alpha(frame_targets_flat, num_classes)
        
        # Compute BBFL-based focal classification loss
        loss_cls = self.focal_loss_fn(pred_logits_flat, frame_targets_flat, alpha=self.focal_alpha)
        loss_expert_probs = self.focal_loss_fn(expert_probs_flat, frame_targets_flat, alpha=self.focal_alpha)
        
        # Compute diversity loss for expert specialization
        loss_div = self.diversity_loss(expert_probs)
        
        # Compute contrastive learning loss between frames and text embeddings
        loss_contrast = self.contrastive_learning(frame_embs, proj_text, frame_targets)
        loss_contrast_text = self.contrastive_text_loss(proj_text)

        loss_contrast2 = self.contrastive_learning(transformed_frames, transformed_text, frame_targets)
        loss_contrast_text2 = self.contrastive_text_loss(transformed_text)
        
        total_loss = loss_cls + self.alpha * loss_div + self.beta * loss_contrast + self.gamma * loss_expert_probs + self.delta * loss_contrast_text + loss_contrast2 + loss_contrast_text2
        loss_dict = {
            "loss_cls": loss_cls,
            "loss_div": loss_div,
            "loss_contrast": loss_contrast,
            "loss_expert_probs": loss_expert_probs,
            "loss_contrast_text": loss_contrast_text,
            "loss_contrast2": loss_contrast2,
            "loss_contrast_text2": loss_contrast_text2
        }
        return total_loss, loss_dict


class AugmentationLoss(nn.Module):
    def __init__(self):
        super(AugmentationLoss, self).__init__()
        
    def forward(self, p, q, softmax=False):
        if softmax:
            p = F.softmax(p, dim=-1)
            q = F.softmax(q, dim=-1)
        p = p.clamp(min=1e-8)
        q = q.clamp(min=1e-8)
        kl_pq = F.kl_div(q.log(), p, reduction="batchmean")  # KL(P || Q)
        kl_qp = F.kl_div(p.log(), q, reduction="batchmean")  # KL(Q || P)

        return (kl_pq + kl_qp) / 2  # Symmetric KL Loss
