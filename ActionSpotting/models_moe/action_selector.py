import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, AutoModel, AutoTokenizer
import os
import math
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import numpy as np

from models_moe.loss import MoeLoss, AugmentationLoss
from models.mstcn import MSTCN2

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        :param d_model: Embedding dimension (D)
        :param max_len: Maximum number of frames (F)
        """
        super(PositionalEncoding, self).__init__()

        # Create a matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # Shape: (max_len, 1)
        
        # Compute the div_term using exponential decay
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as a buffer so it's not updated during training
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: Input tensor of shape (B, F, D)
        :return: Positional encoding added tensor (B, F, D)
        """
        F = x.size(1)  # Extract number of frames
        x = x + self.pe[:F, :].unsqueeze(0)  # Add positional encoding
        return x

class ProxyExpertSelector(nn.Module):
    def __init__(self, frame_dim, device, args):
        super(ProxyExpertSelector, self).__init__()
        self.mstcn = MSTCN2(
            in_dim = frame_dim, 
            num_f_maps = frame_dim, 
            out_dim = frame_dim, 
            num_layers = 10, 
            dropout=0.7)
        self._temperature = args.temperature
        self._device = device
        self._llm_name  = args.llm_name
        self._text_embeddings = None  # Will be initialized later
        self._action_labels = [
            'Background', 
            'Ball out of play', 
            'Goal kick', #Clearance
            'Corner',
            'Direct free-kick', 
            'Foul', 
            'Goal', 
            'Indirect free-kick',
            'Kick-off', 
            'Offside', 
            'Penalty', 
            'Red card',
            'Shots off target', 
            'Shots on target', 
            'Substitution',
            'Throw-in', 
            'Yellow card', 
            'Yellow card to red card']
        self._action_descriptions = [
            "No significant action happening in the play.",
            "The ball has crossed the touchline, stopping the game.",
            "The defending team restarts play from their goal area after the ball crosses the goal line (not a goal) last touched by the attacking team.",
            "The attacking team is awarded a corner kick after the ball crosses the goal line (not a goal) last touched by the defending team.",
            "A set-piece where a player can shoot directly at goal following a foul by the opposing team.",
            "A player commits an illegal action, resulting in a free-kick or penalty for the opposing team.",
            "The ball crosses the goal line between the posts and below the crossbar, awarding a point.",
            "A set-piece where the ball must touch another player before a goal can be scored.",
            "The game restarts from the center circle after a goal is scored or at the beginning of a half.",
            "A player is penalized for being in an offside position when receiving the ball.",
            "A direct free-kick taken from the penalty spot due to a foul inside the penalty area.",
            "A player is sent off the field for a serious foul or two yellow cards, leaving their team with fewer players.",
            "A player attempts a shot that misses the goal frame.",
            "A player takes a shot that is saved by the goalkeeper or results in a goal.",
            "One player is replaced by another from the same team.",
            "A team throws the ball back into play after it crosses the touchline.",
            "A caution is given to a player for misconduct or foul; two result in a red card.",
            "A second yellow card is awarded to a player, leading to an automatic red card and ejection."
        ]
        # Linear projection to align frame and text spaces
        self.text_projection = nn.Sequential(
            nn.Linear(args.text_dim, frame_dim))
        self.frame_projection = nn.Sequential(
            nn.Linear(frame_dim, frame_dim))
            # nn.LeakyReLU())
        self._load_text_data = args.load_text_data
        self._text_data_path = args.text_data_path

    def initialize_text_embeddings(self):
        if self._load_text_data:
            saved_data_path = os.path.join(self._text_data_path, self._llm_name.split('/')[-1].replace(".", "_")  + ".pt")
            if os.path.exists(saved_data_path): 
                self._text_embeddings = torch.load(saved_data_path).to(self._device)
        else:
            llm_model = AutoModel.from_pretrained(self._llm_name).to(self._device)
            tokenizer = AutoTokenizer.from_pretrained(self._llm_name)
            tokenizer.pad_token = tokenizer.eos_token
            target_tokenized = tokenizer(self._action_descriptions, padding = True, truncation = True, return_tensors = 'pt', return_attention_mask = True).to(llm_model.device)
            with torch.no_grad():     
                outputs = llm_model(**target_tokenized) 
                embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings = F.normalize(embeddings, p=2, dim=-1)
            self._text_embeddings = embeddings
            save_path = os.path.join(self._text_data_path, self._llm_name.split('/')[-1].replace(".", "_") + ".pt")
            torch.save(embeddings.detach().cpu(), save_path)
            print(f"Embeddings saved to {save_path} of shape {self._text_embeddings.shape}")

    def forward(self, frame_embeddings):
        """Compute similarity-based routing to experts."""
        text_embeddings = self._text_embeddings.repeat(frame_embeddings.shape[0], 1, 1)
        projected_text = self.text_projection(text_embeddings)  # Align frame-text space
        projected_text = F.normalize(projected_text, p = 2, dim=-1)
        frame_embeddings = F.normalize(frame_embeddings, p = 2, dim=-1)

        # Compute cosine similarity with text embeddings
        similarity_matrix = torch.matmul(frame_embeddings, projected_text.transpose(1, 2))  # (Batch, 30, 18)

        # Soft expert routing using Gumbel-Softmax
        expert_probs = F.gumbel_softmax(similarity_matrix / self._temperature, hard=False, dim=-1)

        return frame_embeddings, expert_probs, projected_text 

class ExpertNetwork(nn.Module):
    def __init__(self, frame_dim, num_classes):
        super(ExpertNetwork, self).__init__()
        self.n1 = nn.Sequential(
            nn.Linear(frame_dim, 1024),
            nn.LeakyReLU())
        self.bn1 = nn.BatchNorm1d(1024)
        self.ln1 = nn.LayerNorm(1024)
        self.n2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU())
        self.bn2 = nn.BatchNorm1d(512)
        self.ln2 = nn.LayerNorm(512)
        self.n3 = nn.Sequential(
            nn.Linear(512, 128),
            nn.LeakyReLU(),
            # nn.Dropout(0.7),
            nn.Linear(128, num_classes)
        )

    def apply_bn(self, x, layer):
        x = x.permute(0, 2, 1)
        if layer == 1:
            x = self.bn1(x)
        elif layer == 2:
            x = self.bn2(x)
        x = x.permute(0, 2, 1)
        return x
    def forward(self, x):
        x = self.n1(x)
        # x = self.apply_bn(x, 1)
        # x = self.ln1(x)
        x = self.n2(x)
        # x = self.apply_bn(x, 2)
        # x = self.ln2(x)
        x = self.n3(x)
        return x

class MoEActionClassifier(nn.Module):
    def __init__(self, frame_dim, num_experts, num_classes):
        super(MoEActionClassifier, self).__init__()
        self._frame_dim = frame_dim
        self._num_experts = num_experts
        self._num_classes = num_classes
        
        # Define 18 expert networks (one per action class)
        self.experts = nn.ModuleList([ExpertNetwork(frame_dim, num_classes) for _ in range(num_experts)])

    def forward(self, frame_embeddings, expert_probs):
        """Compute action class predictions using expert outputs and routing weights."""
        batch_size, num_frames, _ = frame_embeddings.shape
        
        expert_outputs = torch.stack([expert(frame_embeddings) for expert in self.experts], dim=-1)  
        # Shape: (Batch, 30, 18, 18) -> (Batch, Frames, Experts, Action Classes)
        # print(expert_outputs.shape)

        # Weighted sum of expert outputs using routing probabilities
        weighted_output = torch.sum(expert_outputs * expert_probs.unsqueeze(-1), dim=2)
        # weighted_output = expert_outputs * expert_probs 
        # Shape: (Batch, 30, 18) -> Summed over experts

        return weighted_output 

class SoccerActionSelector(nn.Module):
    def __init__(self, args):
        super(SoccerActionSelector, self).__init__()
        self.proxy_selector = ProxyExpertSelector(args.sas.frame_dim, args.device, args.sas.pes)
        self.attention = nn.MultiheadAttention(embed_dim=args.sas.frame_dim, num_heads=8, dropout=0.3, batch_first=True)
        # self.transformer = nn.Transformer(d_model=args.sas.frame_dim, nhead=8, num_encoder_layers=4, num_decoder_layers=4, dim_feedforward=2048, dropout=0.0, batch_first=True)
        self.tf_dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=args.sas.frame_dim, nhead=8, dim_feedforward=2048, dropout=0, batch_first=True), num_layers=4)
        self.tf_dec_text = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=args.sas.frame_dim, nhead=8, dim_feedforward=2048, dropout=0, batch_first=True), num_layers=4)
        self.frame_pos_enc = PositionalEncoding(args.sas.frame_dim, max_len=args.dataset.video_length)
        self.mstcn = MSTCN2(
            in_dim = args.sas.frame_dim, 
            num_f_maps = args.sas.frame_dim, 
            out_dim = args.sas.frame_dim, 
            num_layers = 2, 
            dropout=0)
        self.moe_classifier = MoEActionClassifier(args.sas.frame_dim, args.sas.num_experts,  args.sas.num_classes)
        # self.bn = nn.BatchNorm1d(args.sas.frame_dim)

        # Initialize text embeddings for proxy-based selection
        self.proxy_selector.initialize_text_embeddings()
        self.moe_loss = MoeLoss(args)
        self.aug_loss = AugmentationLoss()
        self._temperature = 0.1

    def plot_expert_probs(self, expert_probs, labels, name):
        plt.figure(figsize=(10, 6))
        labels = labels[0].detach().cpu().numpy()
        sns.heatmap(expert_probs[0, :, :].detach().cpu().numpy(), cmap="viridis", annot=False, yticklabels=labels)
        plt.title(f"{name}")
        # Save plot
        wandb.log({f"{name}": wandb.Image(plt)})
        plt.close()

    def forward_frame_set(self, frame_set, labels = None, compute_loss = False, plot = False):
        processed_frame_embs, expert_probs, projected_text = self.proxy_selector(frame_set)  # Get expert routing probabilities
        # if plot:
        #     self.plot_expert_probs(expert_probs, labels, name = "expert_probs")
    
        transformed_frames = self.tf_dec(processed_frame_embs, projected_text)
        transformed_text = self.tf_dec_text(projected_text, processed_frame_embs)

        transformed_text = F.normalize(transformed_text, p = 2, dim=-1)
        transformed_frames = F.normalize(transformed_frames, p = 2, dim=-1)
        sim = torch.matmul(transformed_frames, transformed_text.transpose(1, 2))
        probs = F.gumbel_softmax(sim / self._temperature, hard=False, dim=-1)
        
        if plot:
            # self.plot_expert_probs(expert_probs1, labels, name = "tf_frames to text")
            # self.plot_expert_probs(expert_probs2, labels, name = "frames to tf_text")
            self.plot_expert_probs(probs, labels, name = "tf_frames to tf_text")
        # class_logits1 = self.moe_classifier(transformed_frames, expert_probs1)
        # class_logits2 = self.moe_classifier(processed_frame_embs, expert_probs2)
        # class_logits = (class_logits1 + class_logits2) / 2
        class_logits = self.moe_classifier(processed_frame_embs, probs)
        
        if compute_loss:
            loss, loss_dict = self.moe_loss(pred_logits = class_logits, frame_targets = labels, expert_probs = probs, frame_embs = processed_frame_embs, proj_text = projected_text, transformed_frames = transformed_frames, transformed_text = transformed_text)
            return loss, loss_dict, class_logits, probs
        return class_logits  
    def forward(self, frame_embeddings, labels = None, compute_loss = False, plot = False):
        if compute_loss:
            loss1, loss_dict1, class_logits1, probs = self.forward_frame_set(frame_embeddings[:, 1::2, :], labels, compute_loss)
            # loss2, class_logits2, probs = self.forward_frame_set(frame_embeddings[:, ::2, :], labels, compute_loss, plot)
            loss = loss1 
            class_logits = class_logits1
            return loss, loss_dict1, class_logits
        class_logits1 = self.forward_frame_set(frame_embeddings[:, 1::2, :])
        # class_logits2 = self.forward_frame_set(frame_embeddings[:, ::2, :])
        class_logits = class_logits1

        return class_logits