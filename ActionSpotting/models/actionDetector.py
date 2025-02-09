import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
import math

from basic import PositionalEncoding

class ActionDetectionModel(nn.Module):
    def __init__(self, args, num_heads=8, refinement_iters=2):
        super(ActionDetectionModel, self).__init__()
        self.frame_emb_dim = args.frame_emb_dim
        self.num_actions = args.num_action_classes
        self.num_frames = args.num_frames
        self.refinement_iters = refinement_iters

        # Learnable parameters (queries and centroids)
        self.action_queries = nn.Parameter(torch.randn(args.num_action_classes, args.frame_emb_dim))
        self.cluster_centroids = nn.Parameter(torch.randn(args.num_action_classes, args.frame_emb_dim))
        
        # Label embeddings will be provided externally but are projected to d_model
        self.label_embeddings_projector = nn.Linear(label_emb_dim, args.frame_emb_dim)
        
        # Learnable positional encodings for frames and queries
        self.frame_pos_enc = PositionalEncoding(args.frame_emb_dim, max_len=10000)
        self.query_pos_enc = nn.Parameter(torch.randn(args.num_action_classes, args.frame_emb_dim))
        
        # Multihead attention modules for query refinement:
        # Each uses the queries as query and different sources as key/value.
        self.attn_frame = MultiheadAttention(args.frame_emb_dim, num_heads)
        self.attn_centroid = MultiheadAttention(args.frame_emb_dim, num_heads)
        self.attn_label = MultiheadAttention(args.frame_emb_dim, num_heads)
        
        # Enhanced fusion: fuse the outputs of the three attention branches.
        self.fusion_mlp = nn.Sequential(
            nn.Linear(args.frame_emb_dim * 3, args.frame_emb_dim),
            nn.ReLU(),
            nn.Linear(args.frame_emb_dim, args.frame_emb_dim)
        )
        self.fusion_ln = nn.LayerNorm(args.frame_emb_dim)
        
        # Multi-scale temporal modeling:
        # Two parallel convolutional branches with different dilation factors.
        self.temporal_conv1 = nn.Conv1d(args.frame_emb_dim, args.frame_emb_dim, kernel_size=3, padding=1, dilation=1)
        self.temporal_conv2 = nn.Conv1d(args.frame_emb_dim, args.frame_emb_dim, kernel_size=3, padding=2, dilation=2)
        self.temporal_fusion = nn.Sequential(
            nn.Linear(args.frame_emb_dim * 2, args.frame_emb_dim),
            nn.ReLU(),
            nn.LayerNorm(args.frame_emb_dim)
        )
        # Self-attention for further temporal context refinement.
        self.temporal_self_attn = MultiheadAttention(args.frame_emb_dim, num_heads)
        
        # Prediction Heads:
        # Classifier: each query predicts one of (num_actions + background) classes.
        self.classifier = nn.Linear(d_model, args.num_action_classes)
        # Regressor: each query predicts normalized start and end times.
        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2)
        )
        
        # Prediction feedback loss (KL divergence)
        self.kl_div_loss = nn.KLDivLoss(reduction='batchmean')
        
    def forward(self, frame_embeddings, action_labels):
        """
        frame_embeddings: Tensor of shape (num_frames, batch_size, d_model)
        action_labels: Precomputed embeddings for action labels (num_actions, d_model)
        """
        batch_size = frame_embeddings.shape[1]
        
        # ---- Positional Encoding ----
        # Add learnable positional encoding to frame embeddings.
        frame_embeddings = frame_embeddings + self.frame_pos_enc.unsqueeze(1)  # (num_frames, batch_size, d_model)
        # Initialize queries with learnable action_queries + query positional encoding.
        queries = (self.action_queries + self.query_pos_enc).unsqueeze(1).expand(-1, batch_size, -1)  # (num_queries, batch_size, d_model)
        
        # Project and expand label embeddings.
        label_embeddings = self.label_embeddings_projector(action_labels)  # (num_actions, d_model)
        label_embeddings = label_embeddings.unsqueeze(1).expand(-1, batch_size, -1)  # (num_actions, batch_size, d_model)
        
        # ---- Iterative Query Refinement ----
        # At each iteration, attend to frame features, cluster centroids, and label embeddings,
        # then fuse the outputs via an MLP with residual connection.
        for i in range(self.refinement_iters):
            # Frame-based cross-attention:
            # Use queries as queries and frame_embeddings as key/value.
            frame_attn_out, _ = self.attn_frame(queries, frame_embeddings, frame_embeddings)
            
            # Centroid-based cross-attention:
            # Expand cluster centroids for the batch.
            centroids = self.cluster_centroids.unsqueeze(1).expand(-1, batch_size, -1)
            centroid_attn_out, _ = self.attn_centroid(queries, centroids, centroids)
            
            # Label-based cross-attention:
            # Use label embeddings as key/value.
            label_attn_out, _ = self.attn_label(queries, label_embeddings, label_embeddings)
            
            # Concatenate the outputs along the feature dimension.
            concat_features = torch.cat([frame_attn_out, centroid_attn_out, label_attn_out], dim=-1)
            refined = self.fusion_mlp(concat_features)
            # Residual connection and normalization.
            queries = self.fusion_ln(queries + refined)
        
        # ---- Multi-Scale Temporal Modeling ----
        # Process frame embeddings with multi-scale convolutions.
        # First, change shape to (batch_size, d_model, num_frames) for conv1d.
        frame_feats = frame_embeddings.permute(1, 2, 0)
        t1 = self.temporal_conv1(frame_feats)  # (batch_size, d_model, num_frames)
        t2 = self.temporal_conv2(frame_feats)  # (batch_size, d_model, num_frames)
        # Fuse the two temporal streams.
        temporal_concat = torch.cat([t1, t2], dim=1)  # (batch_size, 2*d_model, num_frames)
        temporal_fused = self.temporal_fusion(temporal_concat.permute(0, 2, 1))  # (batch_size, num_frames, d_model)
        temporal_fused = temporal_fused.permute(1, 0, 2)  # (num_frames, batch_size, d_model)
        # Self-attention on the aggregated temporal features.
        temp_features, _ = self.temporal_self_attn(temporal_fused, temporal_fused, temporal_fused)
        # Aggregate the temporal features (e.g., via average pooling) to obtain a global feature.
        global_frame_feature = temp_features.mean(dim=0, keepdim=True)  # (1, batch_size, d_model)
        
        # Fuse global frame context into each query.
        queries = queries + global_frame_feature
        
        # ---- Prediction Heads ----
        # Transpose queries to (batch_size, num_queries, d_model) for per–sample predictions.
        queries = queries.transpose(0, 1)
        logits = self.classifier(queries)             # (batch_size, num_queries, num_actions+1)
        boundaries = self.regressor(queries)            # (batch_size, num_queries, 2) – normalized start/end
        
        return logits, boundaries
    
    def compute_prediction_feedback_loss(self, predicted_intervals, attention_maps):
        """
        Aligns attention maps with predicted action intervals using KL divergence.
        predicted_intervals: Tensor of shape (N, 2) with interval boundaries.
        attention_maps: Attention distributions as probabilities that should roughly match IoU similarities.
        """
        IoU_matrix = self.compute_IoU_matrix(predicted_intervals)
        return self.kl_div_loss(attention_maps.log(), IoU_matrix)
    
    def compute_IoU_matrix(self, predicted_intervals):
        """
        Compute an IoU (Intersection over Union) matrix for predicted intervals in a vectorized manner.
        predicted_intervals: Tensor of shape (N, 2) with each row as (start, end).
        Returns a matrix of shape (N, N).
        """
        # predicted_intervals: (N, 2)
        starts = predicted_intervals[:, 0].unsqueeze(1)  # (N, 1)
        ends = predicted_intervals[:, 1].unsqueeze(1)      # (N, 1)
        
        # Compute intersections (broadcasted)
        inter_start = torch.max(starts, starts.transpose(0, 1))
        inter_end = torch.min(ends, ends.transpose(0, 1))
        intersection = torch.clamp(inter_end - inter_start, min=0)
        
        # Compute unions (broadcasted)
        union_start = torch.min(starts, starts.transpose(0, 1))
        union_end = torch.max(ends, ends.transpose(0, 1))
        union = torch.clamp(union_end - union_start, min=1e-6)
        
        iou = intersection / union
        return iou + 1e-6  # Prevent logarithm of zero in KL loss
