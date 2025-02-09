import os
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm  

from utils.helpers import load_json  

class ActionSpotting_v2(Dataset):
    def __init__(self, feature_path, label_path, split, video_length=15, overlap=5, load_data = False):
        self._feature_path = feature_path
        self._label_path = label_path
        self._split = split
        self._video_length = video_length
        self._overlap = overlap
        self._data = []
        self._label_values = {
            'Background': 0, 'Ball out of play': 1, 'Clearance': 2, 'Corner': 3,
            'Direct free-kick': 4, 'Foul': 5, 'Goal': 6, 'Indirect free-kick': 7,
            'Kick-off': 8, 'Offside': 9, 'Penalty': 10, 'Red card': 11,
            'Shots off target': 12, 'Shots on target': 13, 'Substitution': 14,
            'Throw-in': 15, 'Yellow card': 16, 'Yellow->red card': 17
        }
        if load_data:
            saved_data_path = os.apth.join(feature_path,f"ActionSpotting_v2_{split}_video_length_{video_length}_overlap_{overlap}.pt")
            if os.path.exists(saved_data_path): 
                self._data = torch.load(saved_data_path)
        else:
            self.process_dataset()

    def process_dataset(self):
        leagues = os.listdir(os.path.join(self._label_path, self._split))
        
        for league in tqdm(leagues, desc="Processing Leagues"):
            seasons = os.listdir(os.path.join(self._label_path, self._split, league))
            
            for season in tqdm(seasons, desc=f"Processing {league}"):
                games = os.listdir(os.path.join(self._label_path, self._split, league, season))
                
                for game in tqdm(games, desc=f"Processing {league}/{season}"):
                    label_file = os.path.join(self._label_path, self._split, league, season, game, 'Labels-v2.json')
                    
                    for half in range(1, 3):
                        if self._feature_path.split('/')[-1] == 'ResNET_features':
                            feature_file = os.path.join(self._feature_path, league, season, game, f'{half}_ResNET_TF2.npy')
                            self.extract_features_labels(feature_file, label_file, half)

    def get_gametime_half(self, game_time):
        return int(game_time.split()[0])

    def convert_game_info_to_list(self, game_info, half, length_of_half):
        labels = [0] * length_of_half 
        for action in game_info['annotations']:
            if self.get_gametime_half(action['gameTime']) == half:
                position = int(action['position']) // 1000
                if position < length_of_half:
                    labels[position] = self._label_values[action['label']]
                else:
                    print(f"Position {position} is greater than the length of the half {length_of_half}. Defaulting to last time spot.")
                    labels[-1] = self._label_values[action['label']]
        return labels

    def extract_features_labels(self, feature_file, label_file, half):
        if not os.path.exists(feature_file) or not os.path.exists(label_file):
            return

        features = np.load(feature_file)
        game_info = load_json(label_file)
        labels = self.convert_game_info_to_list(game_info, half, features.shape[0] // 2)

        for i in range(0, features.shape[0] // 2, self._overlap):
            data = (
                features[2 * i: min(2 * (i + self._video_length), features.shape[0])],
                labels[i: min(i + self._video_length, len(labels))],
                half, i, i + self._video_length
            )
            if data[0].shape[0] == 2 * self._video_length and len(data[1]) == self._video_length:
                self._data.append(data)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        sample = self._data[idx]  
        feature_vector = torch.from_numpy(sample[0]).float()
        label_vector = torch.tensor(sample[1]).long()
        half = sample[2]
        start_time = sample[3]
        end_time = sample[4]
        return (feature_vector, label_vector, half, start_time, end_time)


