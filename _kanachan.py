from typing import List
import re
import torch
from torch import nn
# from config import Config
# from apex.optimizers import FusedLAMB
from kanachan.training.constants import (
    NUM_TYPES_OF_SPARSE_FEATURES, MAX_NUM_ACTIVE_SPARSE_FEATURES,
    NUM_TYPES_OF_PROGRESSION_FEATURES, MAX_LENGTH_OF_PROGRESSION_FEATURES,
    NUM_TYPES_OF_ACTIONS, MAX_NUM_ACTION_CANDIDATES)
# from kanachan.training.bert.encoder import Encoder
# from kanachan.training.bert.phase1.decoder import Decoder
# from kanachan.training.bert.phase1.model import Model
from kanachan.model_loader import load_model


class Kanachan:
    def __init__(self, model_path="./model/model.kanachan"):
        # self.config = Config()
        # self.encoder = Encoder(
        #     position_encoder=self.config.encoder.position_encoder, 
        #     dimension=self.config.encoder.dimension,
        #     num_heads=self.config.encoder.num_heads, 
        #     dim_feedforward=self.config.encoder.dim_feedforward,
        #     activation_function=self.config.encoder.activation_function, 
        #     dropout=self.config.encoder.dropout,
        #     num_layers=self.config.encoder.num_layers, 
        #     checkpointing=self.config.checkpointing,
        #     device=self.config.device.type, 
        #     dtype=self.config.device.dtype
        # )
        # self.decoder = Decoder(
        #     dimension=self.config.encoder.dimension, 
        #     dim_feedforward=self.config.decoder.dim_feedforward,
        #     activation_function=self.config.decoder.activation_function, 
        #     dropout=self.config.decoder.dropout,
        #     num_layers=self.config.decoder.num_layers, 
        #     device=self.config.device.type, 
        #     dtype=self.config.device.dtype
        # )
        # self.model = Model(self.encoder, self.decoder)
        # self.model.to(device=self.config.device.type, dtype=self.config.device.dtype)
        # # self.optimizer = FusedLAMB(self.model.parameters(), lr=self.config.optimizer.learning_rate, eps=self.config.optimizer.epsilon)
        
        # encoder_state_dict = torch.load(self.config.encoder.load_from, map_location='cpu')
        # encoder_new_state_dict = {}
        # for key, value in encoder_state_dict.items():
        #     new_key = re.sub('^module\\.', '', key)
        #     encoder_new_state_dict[new_key] = value
        # self.encoder.load_state_dict(encoder_new_state_dict)
        # if self.config.device.type != 'cpu':
        #     self.encoder.cuda()

        # decoder_state_dict = torch.load(self.config.decoder.load_from, map_location='cpu')
        # decoder_new_state_dict = {}
        # for key, value in decoder_state_dict.items():
        #     new_key = re.sub('^module\\.', '', key)
        #     decoder_new_state_dict[new_key] = value
        # self.decoder.load_state_dict(decoder_new_state_dict)
        # if self.config.device.type != 'cpu':
        #     self.decoder.cuda()

        # model_state_dict = torch.load(self.config.initial_model, map_location='cpu')
        # model_new_state_dict = {}
        # for key, value in model_state_dict.items():
        #     new_key = re.sub('^module\\.', '', key)
        #     model_new_state_dict[new_key] = value
        # self.model.load_state_dict(model_new_state_dict)
        # if self.config.device.type != 'cpu':
        #     self.model.cuda()

        self.model = load_model(model_path, map_location='cpu')
        self.model.to(device='cpu', dtype=torch.float32)
        self.model.eval()

        pass


    def processAnnotation(self, sparse: List[int], numeric: List[float], progression: List[int], candidates: List[int]):
        for i in range(len(sparse), MAX_NUM_ACTIVE_SPARSE_FEATURES):
            sparse.append(NUM_TYPES_OF_SPARSE_FEATURES)
        sparse = torch.tensor(sparse, device='cpu', dtype=torch.int32)
        sparse = torch.unsqueeze(sparse, dim=0)

        for i in range(2, 6):
            numeric[i] *= 0.0001

        numeric = torch.tensor(numeric, device='cpu', dtype=torch.float32)
        numeric = torch.unsqueeze(numeric, dim=0)

        for i in range(len(progression), MAX_LENGTH_OF_PROGRESSION_FEATURES):
            progression.append(NUM_TYPES_OF_PROGRESSION_FEATURES)
        progression = torch.tensor(progression, device='cpu', dtype=torch.int32)
        progression = torch.unsqueeze(progression, dim=0)

        for i in range(len(candidates), MAX_NUM_ACTION_CANDIDATES):
            candidates.append(NUM_TYPES_OF_ACTIONS + 1)
        candidates = torch.tensor(candidates, device='cpu', dtype=torch.int32)
        candidates = torch.unsqueeze(candidates, dim=0)
        
        return [sparse, numeric, progression, candidates]


    def prediction_function(weights: torch.Tensor) -> torch.Tensor:
        assert weights.dim() == 2
        assert weights.size(1) == MAX_NUM_ACTION_CANDIDATES
        weights = nn.Softmax()(weights)
        return torch.argmax(weights, dim=1)
    

    def kanachan(self, annotation: List[torch.Tensor]) -> torch.Tensor:
        weights = self.model(*(annotation[:4]))
        return weights