import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, AutoModel, AutoTokenizer
import os

from models_moe.loss import MoeLoss

class ProxyExpertSelector(nn.Module):
    def __init__(self, frame_dim, device, args):
        super(ProxyExpertSelector, self).__init__()
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
        # Linear projection to align frame and text spaces
        self.text_projection = nn.Linear(args.text_dim, frame_dim)
        self._load_text_data = args.load_text_data
        self._text_data_path = args.text_data_path
        # if args.load_text_data:
        #     saved_data_path = os.path.join(args.text_data_path, args.llm_name.replace(".", "_")  + ".pt")
        #     if os.path.exists(saved_data_path): 
        #         self._text_embeddings = torch.load(saved_data_path)
        # else:
        #     self.generate_sentence_embeddings()

    def initialize_text_embeddings(self):
        if self._load_text_data:
            saved_data_path = os.path.join(self._text_data_path, self._llm_name.split('/')[-1].replace(".", "_")  + ".pt")
            if os.path.exists(saved_data_path): 
                self._text_embeddings = torch.load(saved_data_path).to(self._device)
        else:
            llm_model = AutoModel.from_pretrained(self._llm_name).to(self._device)
            tokenizer = AutoTokenizer.from_pretrained(self._llm_name)
            tokenizer.pad_token = tokenizer.eos_token
            target_tokenized = tokenizer(self._action_labels, padding = True, truncation = True, return_tensors = 'pt', return_attention_mask = True).to(llm_model.device)
            with torch.no_grad():     
                outputs = llm_model(**target_tokenized) 
                embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings = F.normalize(embeddings, p=2, dim=-1)
            self._text_embeddings = embeddings
            save_path = os.path.join(self._text_data_path, self._llm_name.split('/')[-1].replace(".", "_") + ".pt")
            torch.save(embeddings.detach().cpu(), save_path)
            print(f"Embeddings saved to {save_path} of shape {self._text_embeddings.shape}")
    # def initialize_text_embeddings(self, action_labels):
    #     """Encode action class labels into text embeddings."""
    #     device = next(self.parameters()).device
    #     inputs = self.clip_model.tokenizer(action_labels, return_tensors="pt", padding=True).to(device)
    #     with torch.no_grad():
    #         self.text_embeddings = self.clip_model.get_text_features(**inputs)
    #     self.text_embeddings = F.normalize(self.text_embeddings, dim=-1)

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

        return expert_probs, projected_text 

class ExpertNetwork(nn.Module):
    def __init__(self, frame_dim, num_classes):
        super(ExpertNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(frame_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.network(x)

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

        # Weighted sum of expert outputs using routing probabilities
        weighted_output = torch.sum(expert_outputs * expert_probs.unsqueeze(-1), dim=2)  
        # Shape: (Batch, 30, 18) -> Summed over experts

        return weighted_output 

class SoccerActionSelector(nn.Module):
    def __init__(self, args):
        super(SoccerActionSelector, self).__init__()
        self.proxy_selector = ProxyExpertSelector(args.sas.frame_dim, args.device, args.sas.pes)
        self.moe_classifier = MoEActionClassifier(args.sas.frame_dim, args.sas.num_experts,  args.sas.num_classes)

        # Initialize text embeddings for proxy-based selection
        self.proxy_selector.initialize_text_embeddings()
        self.moe_loss = MoeLoss(args)

    def forward(self, frame_embeddings, labels = None, compute_loss = False):
        expert_probs, projected_text = self.proxy_selector(frame_embeddings)  # Get expert routing probabilities
        class_logits = self.moe_classifier(frame_embeddings, expert_probs)  # MoE action classification
        if compute_loss:
            loss = self.moe_loss(pred_logits = class_logits, frame_targets = labels, expert_probs = expert_probs, frame_embs = frame_embeddings, proj_text = projected_text)
            return loss, class_logits
        return class_logits  # Shape: (Batch, 18)