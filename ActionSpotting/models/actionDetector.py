import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
import math
import wandb
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
import numpy as np  
import seaborn as sns

from models.basic import PositionalEncoding, logit2prob, time_mask
from models.blocks import Block, UpdateBlock, InputBlock, UpdateBlockTDU
from utils.helpers import to_numpy
from models.loss import MatchCriterion, torch_class_label_to_segment_label

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


class ActionDetectionModel2(nn.Module):
    def __init__(self, args):
        super(ActionDetectionModel2, self).__init__()
        self.args = args
        self.frame_emb_dim = args.actionModel.frame_emb_dim
        self.num_actions = args.actionModel.num_action_classes

        self.frame_pe = PositionalEncoding(args.actionModel.inputBlock.hid_dim, max_len=10000, empty=(not args.actionModel.fpos) )
        if args.actionModel.use_cmr:
            self.channel_masking_dropout = nn.Dropout2d(p=args.actionModel.cmr)

        self.action_query = nn.Parameter(torch.randn([args.actionModel.num_action_queries, 1, args.actionModel.frame_emb_dim]))
        # self.action_query = nn.Parameter(torch.randn([self.num_actions, 1, args.actionModel.frame_emb_dim]))
        

        # block configuration
        block_list = []
        for i, t in enumerate(args.actionModel.blocks):
            if t == 'i':
                block = InputBlock(args = args, in_dim = args.actionModel.frame_emb_dim, nclass = args.actionModel.num_action_classes)
            elif t == 'u':
                # update_from(cfg.Bu, base_cfg, inplace=True)
                # base_cfg = cfg.Bu
                block = UpdateBlock(args = args, nclass = args.actionModel.num_action_classes)
            elif t == 'U':
                # update_from(cfg.BU, base_cfg, inplace=True)
                # base_cfg = cfg.BU
                block = UpdateBlockTDU(args, args.actionModel.num_action_classes)

            block_list.append(block)

        self.block_list = nn.ModuleList(block_list)

        self.mcriterion = None
        
    def _forward_one_video(self, seq, transcript=None, labels = None, visualization_map = False):
        # prepare frame feature
        frame_feature = seq
        frame_pe = self.frame_pe(seq)
        if self.args.actionModel.use_cmr:
            frame_feature = frame_feature.permute([1, 2, 0])
            frame_feature = self.channel_masking_dropout(frame_feature)
            frame_feature = frame_feature.permute([2, 0, 1])

        if self.args.actionModel.tm.use and self.training:
            frame_feature = time_mask(feature = frame_feature, 
                        T = self.args.actionModel.tm.t, 
                        num_masks = self.args.actionModel.tm.m, 
                        p = self.args.actionModel.tm.p, 
                        replace_with_zero=True)

        # prepare action feature
        # if not self.cfg.FACT.trans:
        action_pe = self.action_query # M, B(=1), H
        action_feature = torch.zeros_like(action_pe)
        # else:
        #     action_pe = self.action_pe(transcript)
        #     action_feature = self.action_embed(transcript).unsqueeze(1)

        #     action_feature = action_feature + action_pe
        #     action_pe = torch.zeros_like(action_pe)

        # forward
        # frame_feature: T, B(=1), H
        # action_feature: M, B(=1), H
        block_output = []
        for i, block in enumerate(self.block_list):
            if visualization_map:
                if i == 0:
                    self.plot_visualization_map(frame_feature, labels, feature_name = 'frame_feature_initial')
                    # self.plot_visualization_map(action_feature, feature_name = 'action_feature_initial')
                else:
                    self.plot_visualization_map(frame_feature, labels, feature_name = f'frame_feature_after_block_{i}')
                    # self.plot_visualization_map(action_feature, feature_name = f'action_feature_after_block_{i}')
            frame_feature, action_feature = block(frame_feature, action_feature, frame_pe, action_pe)
            block_output.append([frame_feature, action_feature])
        return block_output

    def plot_visualization_map(self, embeddings, labels, feature_name = ''):
        embeddings = embeddings.squeeze().detach().cpu().numpy()
        labels = labels.squeeze().cpu().numpy()
        tsne = TSNE(
            n_components=2, perplexity = 2, # Reduce to 2D for visualization
            learning_rate='auto', init='pca', random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)  

        # Plot the 2D representation
        plt.figure(figsize=(8, 6))
        palette = sns.color_palette("hsv", len(set(labels)))  # Generate distinct colors
        sns.scatterplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], hue=labels, palette=palette, s=100, edgecolor='k')
        # plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='blue', edgecolors='k')
        for i, (x, y) in enumerate(embeddings_2d):
            plt.text(x, y, str(i), fontsize=12, ha='right', va='bottom')

        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.title(f"{feature_name}")
        wandb.log({"t-SNE Visualization": wandb.Image(plt)})
        plt.close()

    def _loss_one_video(self, label):
        mcriterion: MatchCriterion = self.mcriterion
        mcriterion.set_label(label)

        block : Block = self.block_list[-1]
        cprob = logit2prob(block.action_clogit, dim=-1)
        match = mcriterion.match(cprob, block.a2f_attn)

        ######## per block loss
        loss_list = []
        for block in self.block_list:
            loss = block.compute_loss(mcriterion, match)
            loss_list.append(loss)

        self.loss_list = loss_list
        final_loss = sum(loss_list) / len(loss_list)
        return final_loss

    def forward(self, seq_list, label_list, compute_loss=False, visualization_map = False):

        save_list = []
        final_loss = []

        for i, (seq, label) in enumerate(zip(seq_list, label_list)):
            if self.frame_emb_dim == 2048:
                seq = seq[::2, :]
            else:
                seq = seq[:, :self.frame_emb_dim]
            seq = seq.unsqueeze(1)
            # print("Seq shape: ", seq.shape)
            trans = torch_class_label_to_segment_label(label)[0]
            if i == 0 and visualization_map:
                self._forward_one_video(seq, trans, labels = label, visualization_map = visualization_map)
            else:
                self._forward_one_video(seq, trans)

            pred = self.block_list[-1].eval(trans)
            save_data = {'pred': to_numpy(pred)}
            save_list.append(save_data)

            if compute_loss:
                loss = self._loss_one_video(label)
                final_loss.append(loss)
                save_data['loss'] = { 'loss': loss.item() }


        if compute_loss:
            final_loss = sum(final_loss) / len(final_loss)
            return final_loss, save_list
        else:
            return save_list

    def save_model(self, fname):
        torch.save(self.state_dict(), fname)
    


