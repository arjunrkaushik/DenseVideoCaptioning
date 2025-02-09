import json
from tqdm import tqdm
import torch 

from utils.dataset import ActionSpotting_v2
def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def save_dataset(feature_path, label_path, split, video_length, overlap):
    dataset = ActionSpotting_v2(feature_path, label_path, split)
    print("\nSaving dataset...")
    data_list = [dataset[i] for i in tqdm(range(len(dataset)), desc="Saving Samples")]
    
    if self._feature_path.split('/')[-1] == 'ResNET_features':
        save_path = f"/data/kaushik3/SoccerData/ResNET_features/ActionSpotting_v2_{split}_video_length_{video_length}_overlap_{overlap}.pt"
    else:
        save_path = f"/data/kaushik3/SoccerData/Baidu_Features/ActionSpotting_v2_{split}_video_length_{video_length}_overlap_{overlap}.pt"
    torch.save(data_list, save_path)
    print(f"Dataset saved to {save_path}")