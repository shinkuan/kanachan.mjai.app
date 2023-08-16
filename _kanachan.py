#!/usr/bin/env python3

import re
from pathlib import Path
from collections import Counter
import json
import sys
from typing import (Optional, List,)
import torch
from mahjong.shanten import Shanten
from hand_calculator import HandCalculator
from kanachan.model_loader import load_model
from kanachan.training.constants import (
    NUM_TYPES_OF_SPARSE_FEATURES, MAX_NUM_ACTIVE_SPARSE_FEATURES,
    NUM_TYPES_OF_PROGRESSION_FEATURES, MAX_LENGTH_OF_PROGRESSION_FEATURES,
    NUM_TYPES_OF_ACTIONS, MAX_NUM_ACTION_CANDIDATES)

from mjai.bot import Bot
from mjai.bot.tools import calc_shanten, convert_vec34_to_short

from loguru import logger


MJAI_VEC34_TILES = [
    "1m",
    "2m",
    "3m",
    "4m",
    "5m",
    "6m",
    "7m",
    "8m",
    "9m",
    "1p",
    "2p",
    "3p",
    "4p",
    "5p",
    "6p",
    "7p",
    "8p",
    "9p",
    "1s",
    "2s",
    "3s",
    "4s",
    "5s",
    "6s",
    "7s",
    "8s",
    "9s",
    "E",
    "S",
    "W",
    "N",
    "P",
    "F",
    "C",
]

VEC34_MJAI_TILES = {}
for i, tile in enumerate(MJAI_VEC34_TILES):
    VEC34_MJAI_TILES[tile] = i

_TILE_OFFSETS = (
      0,   1,   5,   9,  13,  17,  20,  24,  28,  32,
     36,  37,  41,  45,  49,  53,  56,  60,  64,  68,
     72,  73,  77,  81,  85,  89,  92,  96, 100, 104,
    108, 112, 116, 120, 124, 128, 132, 136
)

_NUM2TILE = (
    '5mr', '1m', '2m', '3m', '4m', '5m', '6m', '7m', '8m', '9m',
    '5pr', '1p', '2p', '3p', '4p', '5p', '6p', '7p', '8p', '9p',
    '5sr', '1s', '2s', '3s', '4s', '5s', '6s', '7s', '8s', '9s',
    'E', 'S', 'W', 'N', 'P', 'F', 'C'
)

_TILE2NUM = {}
for i, tile in enumerate(_NUM2TILE):
    _TILE2NUM[tile] = i

_TILE34TILE37 = (
     1, 2, 3, 4, 5, 6, 7, 8, 9,
    11,12,13,14,15,16,17,18,19,
    21,22,23,24,25,26,27,28,29,
    30,31,32,33,34,35,36
)

_TILE37TILE34 = (
     4, 0, 1, 2, 3, 4, 5, 6, 7, 8,
    13, 9,10,11,12,13,14,15,16,17,
    22,18,19,20,21,22,23,24,25,26,
    27,28,29,30,31,32,33
)

_NUM2CHI = (
    ('1m',  ['2m', '3m']),
    ('2m',  ['1m', '3m']),
    ('2m',  ['3m', '4m']),
    ('3m',  ['1m', '2m']),
    ('3m',  ['2m', '4m']),
    ('3m',  ['4m', '5m']),
    ('3m',  ['4m', '5mr']),
    ('4m',  ['2m', '3m']),
    ('4m',  ['3m', '5m']),
    ('4m',  ['3m', '5mr']),
    ('4m',  ['5m', '6m']),
    ('4m',  ['5mr', '6m']),
    ('5m',  ['3m', '4m']),
    ('5mr', ['3m', '4m']),
    ('5m',  ['4m', '6m']),
    ('5mr', ['4m', '6m']),
    ('5m',  ['6m', '7m']),
    ('5mr', ['6m', '7m']),
    ('6m',  ['4m', '5m']),
    ('6m',  ['4m', '5mr']),
    ('6m',  ['5m', '7m']),
    ('6m',  ['5mr', '7m']),
    ('6m',  ['7m', '8m']),
    ('7m',  ['5m', '6m']),
    ('7m',  ['5mr', '6m']),
    ('7m',  ['6m', '8m']),
    ('7m',  ['8m', '9m']),
    ('8m',  ['6m', '7m']),
    ('8m',  ['7m', '9m']),
    ('9m',  ['7m', '8m']),
    ('1p',  ['2p', '3p']),
    ('2p',  ['1p', '3p']),
    ('2p',  ['3p', '4p']),
    ('3p',  ['1p', '2p']),
    ('3p',  ['2p', '4p']),
    ('3p',  ['4p', '5p']),
    ('3p',  ['4p', '5pr']),
    ('4p',  ['2p', '3p']),
    ('4p',  ['3p', '5p']),
    ('4p',  ['3p', '5pr']),
    ('4p',  ['5p', '6p']),
    ('4p',  ['5pr', '6p']),
    ('5p',  ['3p', '4p']),
    ('5pr', ['3p', '4p']),
    ('5p',  ['4p', '6p']),
    ('5pr', ['4p', '6p']),
    ('5p',  ['6p', '7p']),
    ('5pr', ['6p', '7p']),
    ('6p',  ['4p', '5p']),
    ('6p',  ['4p', '5pr']),
    ('6p',  ['5p', '7p']),
    ('6p',  ['5pr', '7p']),
    ('6p',  ['7p', '8p']),
    ('7p',  ['5p', '6p']),
    ('7p',  ['5pr', '6p']),
    ('7p',  ['6p', '8p']),
    ('7p',  ['8p', '9p']),
    ('8p',  ['6p', '7p']),
    ('8p',  ['7p', '9p']),
    ('9p',  ['7p', '8p']),
    ('1s',  ['2s', '3s']),
    ('2s',  ['1s', '3s']),
    ('2s',  ['3s', '4s']),
    ('3s',  ['1s', '2s']),
    ('3s',  ['2s', '4s']),
    ('3s',  ['4s', '5s']),
    ('3s',  ['4s', '5sr']),
    ('4s',  ['2s', '3s']),
    ('4s',  ['3s', '5s']),
    ('4s',  ['3s', '5sr']),
    ('4s',  ['5s', '6s']),
    ('4s',  ['5sr', '6s']),
    ('5s',  ['3s', '4s']),
    ('5sr', ['3s', '4s']),
    ('5s',  ['4s', '6s']),
    ('5sr', ['4s', '6s']),
    ('5s',  ['6s', '7s']),
    ('5sr', ['6s', '7s']),
    ('6s',  ['4s', '5s']),
    ('6s',  ['4s', '5sr']),
    ('6s',  ['5s', '7s']),
    ('6s',  ['5sr', '7s']),
    ('6s',  ['7s', '8s']),
    ('7s',  ['5s', '6s']),
    ('7s',  ['5sr', '6s']),
    ('7s',  ['6s', '8s']),
    ('7s',  ['8s', '9s']),
    ('8s',  ['6s', '7s']),
    ('8s',  ['7s', '9s']),
    ('9s',  ['7s', '8s'])
)

_CHI2NUM = {}
for i, (tile, consumed) in enumerate(_NUM2CHI):
    k = (tile, tuple(consumed))
    _CHI2NUM[k] = i

_NUM2PENG = (
    ('1m',  ['1m', '1m']),
    ('2m',  ['2m', '2m']),
    ('3m',  ['3m', '3m']),
    ('4m',  ['4m', '4m']),
    ('5m',  ['5m', '5m']),
    ('5m',  ['5mr', '5m']),
    ('5mr', ['5m', '5m']),
    ('6m',  ['6m', '6m']),
    ('7m',  ['7m', '7m']),
    ('8m',  ['8m', '8m']),
    ('9m',  ['9m', '9m']),
    ('1p',  ['1p', '1p']),
    ('2p',  ['2p', '2p']),
    ('3p',  ['3p', '3p']),
    ('4p',  ['4p', '4p']),
    ('5p',  ['5p', '5p']),
    ('5p',  ['5pr', '5p']),
    ('5pr', ['5p', '5p']),
    ('6p',  ['6p', '6p']),
    ('7p',  ['7p', '7p']),
    ('8p',  ['8p', '8p']),
    ('9p',  ['9p', '9p']),
    ('1s',  ['1s', '1s']),
    ('2s',  ['2s', '2s']),
    ('3s',  ['3s', '3s']),
    ('4s',  ['4s', '4s']),
    ('5s',  ['5s', '5s']),
    ('5s',  ['5sr', '5s']),
    ('5sr', ['5s', '5s']),
    ('6s',  ['6s', '6s']),
    ('7s',  ['7s', '7s']),
    ('8s',  ['8s', '8s']),
    ('9s',  ['9s', '9s']),
    ('E',   ['E', 'E']),
    ('S',   ['S', 'S']),
    ('W',   ['W', 'W']),
    ('N',   ['N', 'N']),
    ('P',   ['P', 'P']),
    ('F',   ['F', 'F']),
    ('C',   ['C', 'C'])
)

_PENG2NUM = {}
for i, (tile, consumed) in enumerate(_NUM2PENG):
    k = (tile, tuple(consumed))
    _PENG2NUM[k] = i

_NUM2DAMINGGANG = (
    ('5mr', ['5m', '5m', '5m']),
    ('1m',  ['1m', '1m', '1m']),
    ('2m',  ['2m', '2m', '2m']),
    ('3m',  ['3m', '3m', '3m']),
    ('4m',  ['4m', '4m', '4m']),
    ('5m',  ['5mr', '5m', '5m']),
    ('6m',  ['6m', '6m', '6m']),
    ('7m',  ['7m', '7m', '7m']),
    ('8m',  ['8m', '8m', '8m']),
    ('9m',  ['9m', '9m', '9m']),
    ('5pr', ['5p', '5p', '5p']),
    ('1p',  ['1p', '1p', '1p']),
    ('2p',  ['2p', '2p', '2p']),
    ('3p',  ['3p', '3p', '3p']),
    ('4p',  ['4p', '4p', '4p']),
    ('5p',  ['5pr', '5p', '5p']),
    ('6p',  ['6p', '6p', '6p']),
    ('7p',  ['7p', '7p', '7p']),
    ('8p',  ['8p', '8p', '8p']),
    ('9p',  ['9p', '9p', '9p']),
    ('5sr', ['5s', '5s', '5s']),
    ('1s',  ['1s', '1s', '1s']),
    ('2s',  ['2s', '2s', '2s']),
    ('3s',  ['3s', '3s', '3s']),
    ('4s',  ['4s', '4s', '4s']),
    ('5s',  ['5sr', '5s', '5s']),
    ('6s',  ['6s', '6s', '6s']),
    ('7s',  ['7s', '7s', '7s']),
    ('8s',  ['8s', '8s', '8s']),
    ('9s',  ['9s', '9s', '9s']),
    ('E',   ['E', 'E', 'E']),
    ('S',   ['S', 'S', 'S']),
    ('W',   ['W', 'W', 'W']),
    ('N',   ['N', 'N', 'N']),
    ('P',   ['P', 'P', 'P']),
    ('F',   ['F', 'F', 'F']),
    ('C',   ['C', 'C', 'C'])
)

_DAMINGGANG2NUM = {}
for i, (tile, consumed) in enumerate(_NUM2DAMINGGANG):
    k = (tile, tuple(consumed))
    _DAMINGGANG2NUM[k] = i

_NUM2ANGANG = (
    ['1m', '1m', '1m', '1m'],
    ['2m', '2m', '2m', '2m'],
    ['3m', '3m', '3m', '3m'],
    ['4m', '4m', '4m', '4m'],
    ['5mr', '5m', '5m', '5m'],
    ['6m', '6m', '6m', '6m'],
    ['7m', '7m', '7m', '7m'],
    ['8m', '8m', '8m', '8m'],
    ['9m', '9m', '9m', '9m'],
    ['1p', '1p', '1p', '1p'],
    ['2p', '2p', '2p', '2p'],
    ['3p', '3p', '3p', '3p'],
    ['4p', '4p', '4p', '4p'],
    ['5pr', '5p', '5p', '5p'],
    ['6p', '6p', '6p', '6p'],
    ['7p', '7p', '7p', '7p'],
    ['8p', '8p', '8p', '8p'],
    ['9p', '9p', '9p', '9p'],
    ['1s', '1s', '1s', '1s'],
    ['2s', '2s', '2s', '2s'],
    ['3s', '3s', '3s', '3s'],
    ['4s', '4s', '4s', '4s'],
    ['5sr', '5s', '5s', '5s'],
    ['6s', '6s', '6s', '6s'],
    ['7s', '7s', '7s', '7s'],
    ['8s', '8s', '8s', '8s'],
    ['9s', '9s', '9s', '9s'],
    ['E', 'E', 'E', 'E'],
    ['S', 'S', 'S', 'S'],
    ['W', 'W', 'W', 'W'],
    ['N', 'N', 'N', 'N'],
    ['P', 'P', 'P', 'P'],
    ['F', 'F', 'F', 'F'],
    ['C', 'C', 'C', 'C']
)

_ANGANG2NUM = {}
for i, consumed in enumerate(_NUM2ANGANG):
    k = tuple(consumed)
    _ANGANG2NUM[k] = i

_NUM2JIAGANG = (
    ('5mr', ['5m', '5m', '5m']),
    ('1m',  ['1m', '1m', '1m']),
    ('2m',  ['2m', '2m', '2m']),
    ('3m',  ['3m', '3m', '3m']),
    ('4m',  ['4m', '4m', '4m']),
    ('5m',  ['5mr', '5m', '5m']),
    ('6m',  ['6m', '6m', '6m']),
    ('7m',  ['7m', '7m', '7m']),
    ('8m',  ['8m', '8m', '8m']),
    ('9m',  ['9m', '9m', '9m']),
    ('5pr', ['5p', '5p', '5p']),
    ('1p',  ['1p', '1p', '1p']),
    ('2p',  ['2p', '2p', '2p']),
    ('3p',  ['3p', '3p', '3p']),
    ('4p',  ['4p', '4p', '4p']),
    ('5p',  ['5pr', '5p', '5p']),
    ('6p',  ['6p', '6p', '6p']),
    ('7p',  ['7p', '7p', '7p']),
    ('8p',  ['8p', '8p', '8p']),
    ('9p',  ['9p', '9p', '9p']),
    ('5sr', ['5s', '5s', '5s']),
    ('1s',  ['1s', '1s', '1s']),
    ('2s',  ['2s', '2s', '2s']),
    ('3s',  ['3s', '3s', '3s']),
    ('4s',  ['4s', '4s', '4s']),
    ('5s',  ['5sr', '5s', '5s']),
    ('6s',  ['6s', '6s', '6s']),
    ('7s',  ['7s', '7s', '7s']),
    ('8s',  ['8s', '8s', '8s']),
    ('9s',  ['9s', '9s', '9s']),
    ('E',   ['E', 'E', 'E']),
    ('S',   ['S', 'S', 'S']),
    ('W',   ['W', 'W', 'W']),
    ('N',   ['N', 'N', 'N']),
    ('P',   ['P', 'P', 'P']),
    ('F',   ['F', 'F', 'F']),
    ('C',   ['C', 'C', 'C'])
)

class KanachanBot(Bot):
    def __init__(self, player_id: int = 0, model_path="./model/model.kanachan"):
        super().__init__(player_id)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__model = load_model(model_path, map_location=self.device)
        self.__model.to(device=self.device, dtype=torch.float32)
        self.__model.eval()

        with open('./game.json', encoding='UTF-8') as f:
            game_config = json.load(f)
            self.__game_state = GameState(
                my_name=game_config['my_name'], 
                room=game_config['room'], 
                game_style=game_config['game_style'], 
                my_grade=game_config['my_grade'], 
                opponent_grade=game_config['opponent_grade']
            )
            self.__game_state.on_new_round(player_id, [25000, 25000, 25000, 25000])

        self.__progression = []
        self.__left_tile = 70
        self.__liqi_to_be_accepted = [False] * 4
        self.__liqi_message = ""

    def think(self) -> str:
        if self.__liqi_to_be_accepted[self.player_id]:
            return self.__liqi_message
        if self.__left_tile == 70:
            return self.action_nothing()

        seat = self.player_id

        self.__game_state.__player_scores = self.scores
        sparse = []
        sparse.append(self.__game_state.get_room())
        sparse.append(self.__game_state.get_game_style() + 5)
        sparse.append(seat + 7)
        sparse.append(self.player_state.bakaze - 27 + 11)
        sparse.append(self.kyoku + 14)
        for i, dora_indicator in self.dora_indicators:
            sparse.append(_TILE2NUM[dora_indicator] + 37 * i + 18)
        assert(self.__left_tile != 70)
        sparse.append(self.__left_tile + 203)
        sparse.append(self.__game_state.get_player_grade(seat) + 273)
        sparse.append(self.__game_state.get_player_rank(seat) + 289)
        sparse.append(self.__game_state.get_player_grade((seat + 1) % 4) + 293)
        sparse.append(self.__game_state.get_player_rank((seat + 1) % 4) + 309)
        sparse.append(self.__game_state.get_player_grade((seat + 2) % 4) + 313)
        sparse.append(self.__game_state.get_player_rank((seat + 2) % 4) + 329)
        sparse.append(self.__game_state.get_player_grade((seat + 3) % 4) + 333)
        sparse.append(self.__game_state.get_player_rank((seat + 3) % 4) + 349)

        hand_encode = [None] * 136
        hand34 = self.tehai_vec34
        if self.akas_in_hand[0]:
            hand34[4] -= 1
            hand_encode[0] = 1
        if self.akas_in_hand[1]:
            hand34[13] -= 1
            hand_encode[36] = 1
        if self.akas_in_hand[2]:
            hand34[22] -= 1
            hand_encode[72] = 1
        for tile_idx, tile_count in enumerate(hand34):
            tile37 = _TILE34TILE37[tile_idx]
            for i in range(_TILE_OFFSETS[tile37], _TILE_OFFSETS[tile37] + tile_count):
                hand_encode[i] = 1
        for i in range(136):
            if hand_encode[i] == 1:
                sparse.append(i + 353)

        if self.last_self_tsumo != "":
            zimo_tile = _TILE2NUM[self.last_self_tsumo]
            sparse.append(zimo_tile + 489)
        for i in range(len(sparse), MAX_NUM_ACTIVE_SPARSE_FEATURES):
            sparse.append(NUM_TYPES_OF_SPARSE_FEATURES)
        sparse = torch.tensor(sparse, device=self.device, dtype=torch.int32)
        sparse = torch.unsqueeze(sparse, dim=0)

        numeric = []
        numeric.append(self.honba)
        numeric.append(self.kyotaku)
        numeric.append(self.__game_state.get_player_score(seat) / 10000.0)
        numeric.append(self.__game_state.get_player_score((seat + 1) % 4) / 10000.0)
        numeric.append(self.__game_state.get_player_score((seat + 2) % 4) / 10000.0)
        numeric.append(self.__game_state.get_player_score((seat + 3) % 4) / 10000.0)
        numeric = torch.tensor(numeric, device=self.device, dtype=torch.float32)
        numeric = torch.unsqueeze(numeric, dim=0)

        progression = self.__progression.copy()
        for i in range(len(progression), MAX_LENGTH_OF_PROGRESSION_FEATURES):
            progression.append(NUM_TYPES_OF_PROGRESSION_FEATURES)
        progression = torch.tensor(progression, device='cpu', dtype=torch.int32)
        progression = torch.unsqueeze(progression, dim=0)

        candidates = self.get_candidates()
        candidates_ = list(candidates)
        for i in range(len(candidates_), MAX_NUM_ACTION_CANDIDATES):
            candidates_.append(NUM_TYPES_OF_ACTIONS + 1)
        candidates_ = torch.tensor(candidates_, device='cpu', dtype=torch.int32)
        candidates_ = torch.unsqueeze(candidates_, dim=0)

        with torch.no_grad():
            prediction = self.__model(sparse, numeric, progression, candidates_)
            prediction = torch.squeeze(prediction, dim=0)
            prediction = prediction[:len(candidates)]
            argmax = torch.argmax(prediction)
            argmax = argmax.item()
        candidates_ = torch.squeeze(candidates_, dim=0)
        decision = candidates_[argmax].item()

        logger.info(
            f"{self.bakaze}{self.kyoku}-{self.honba}: {[sparse, numeric, progression, candidates_]}, DECICISION={decision}"
        )

        if 0 <= decision and decision <= 147:
            tile = decision // 4
            tile = _NUM2TILE[tile]
            encode = decision % 4
            moqi = encode // 2 == 1
            encode = encode % 2
            liqi = encode == 1

            if liqi:
                response = self.action_riichi()
                self.__liqi_message = self.action_discard(tile)
                return response
            else:
                return self.action_discard(tile)

        if 148 <= decision and decision <= 181:
            angang = _NUM2ANGANG[decision - 148]
            return self.action_ankan(angang)

        if 182 <= decision and decision <= 218:
            tile, consumed = _NUM2JIAGANG[decision - 182]
            return self.action_kakan(consumed)# TODO Has problem now.

        if decision == 219:
            return self.action_tsumo_agari()

        if decision == 220:
            return self.action_ryukyoku()# TODO Not Impl yet

        if decision == 221:
            return self.action_nothing()

        if 222 <= decision and decision <= 311:
            tile, consumed = _NUM2CHI[decision - 222]
            return self.action_chi(consumed)

        if 312 <= decision and decision <= 431:
            encode = decision - 312
            relseat = encode // 40
            target = (seat + relseat + 1) % 4
            encode = encode % 40
            tile, consumed = _NUM2PENG[encode]
            return self.action_pon(consumed)

        if 432 <= decision and decision <= 542:
            encode = decision - 432
            relseat = encode // 37
            target = (seat + relseat + 1) % 4
            encode = encode % 37
            tile, consumed = _NUM2DAMINGGANG[encode]
            return self.action_daiminkan(consumed)

        if 543 <= decision and decision <= 545:
            return self.action_ron_agari()

        raise RuntimeError(f'An invalid decision (decision = {decision}).')
    
    #OVERRIDE to pass event to progression parser
    def react(self, input_str: str) -> str:
        # try:
        events = json.loads(input_str)
        if len(events) == 0:
            raise ValueError("Empty events")
        for event in events:
            if event["type"] == "start_game":
                self.__discard_events = []
                self.__call_events = []
                self.__dora_indicators = []
            if event["type"] == "start_kyoku" or event["type"] == "dora":
                self.__dora_indicators.append(event["dora_marker"])
            if event["type"] == "dahai":
                self.__discard_events.append(event)
            if event["type"] in [
                "chi",
                "pon",
                "daiminkan",
                "kakan",
                "ankan",
            ]:
                self.__call_events.append(event)
            self.event_progression_parser(event)

            self.action_candidate = self.player_state.update(
                json.dumps(event)
            )

        resp = self.think()
        return resp

        # except Exception as e:
        #     print(
        #         "===========================================", file=sys.stderr
        #     )
        #     print(f"Exception: {str(e)}", file=sys.stderr)
        #     print("Brief info:", file=sys.stderr)
        #     print(self.brief_info(), file=sys.stderr)
        #     print("", file=sys.stderr)

        return json.dumps({"type": "none"}, separators=(",", ":"))
    
    def event_progression_parser(self, event):
        if event["type"] == "start_game":
            self.player_id =event["id"]
        if event["type"] == "start_kyoku":
            self.__left_tile = 70
            self.__progression = [0]
            self.__liqi_to_be_accepted = [False] * 4
        if event["type"] == "tsumo":
            self.__left_tile -= 1
        if event["type"] == "dahai":
            moqi = event["tsumogiri"]
            if self.__left_tile == 69:
                moqi = False
            liqi = self.__liqi_to_be_accepted[event["actor"]]
            self.__liqi_to_be_accepted[event["actor"]] = False
            self.__progression.append(5 + event["actor"] * 148 + _TILE2NUM[event["pai"]] * 4 + (2 if moqi else 0) + (1 if liqi else 0))
        if event["type"] == "reach":
            self.__liqi_to_be_accepted[event["actor"]] = True
        if event["type"] == "chi":
            chi = (event["pai"], tuple(event["consumed"]))
            if chi not in _CHI2NUM:
                raise RuntimeError(chi)
            chi = _CHI2NUM[chi]
            self.__progression.append(597 + event["actor"] * 90 + chi)
        if event["type"] == "pon":
            pon = (event["pai"], tuple(event["consumed"]))
            if pon not in _PENG2NUM:
                raise RuntimeError(pon)
            pon = _PENG2NUM[pon]
            relseat = ((event["target"] - event["actor"] + 4) % 4) - 1
            self.__progression.append(957 + event["actor"] * 120 + relseat * 40 + pon)
        if event["type"] == "daiminkan":
            daminggang = (event["pai"], tuple(event["consumed"]))
            if daminggang not in _DAMINGGANG2NUM:
                raise RuntimeError(daminggang)
            daminggang = _DAMINGGANG2NUM[daminggang]
            relseat = ((event["target"] - event["actor"] + 4) % 4) - 1
            self.__progression.append(1437 + event["actor"] * 111 + relseat * 37 + daminggang)
        if event["type"] == "ankan":
            angang = tuple(event["consumed"])
            if angang not in _ANGANG2NUM:
                raise RuntimeError(angang)
            angang = _ANGANG2NUM[angang]
            self.__progression.append(1881 + event["actor"] * 34 + angang)
        if event["type"] == "kakan":
            self.__progression.append(2017 + event["actor"] * 37 + _TILE2NUM[event["pai"]])

    def get_candidates(self):
        candidates = set()
        if self.can_discard:
            hand37_combine = [0]*37
            hand34_combine = self.tehai_vec34
            for idx, pai_num in enumerate(self.tehai_vec34):
                hand37_combine[_TILE34TILE37[idx]] = pai_num
            if self.akas_in_hand[0]:
                hand37_combine[0] = 1
                hand37_combine[5] -=1
            if self.akas_in_hand[1]:
                hand37_combine[10] = 1
                hand37_combine[15] -=1
            if self.akas_in_hand[2]:
                hand37_combine[20] = 1
                hand37_combine[25] -=1
            hand37 = hand37_combine.copy()
            hand34 = hand34_combine.copy()
            isMenzenchin = len(self.get_call_events(self.player_id)) == 0
            if self.last_self_tsumo != "":
                tsumo_tile37 = _TILE2NUM[self.last_self_tsumo]
                hand37[_TILE2NUM[self.last_self_tsumo]]-=1
                hand34[VEC34_MJAI_TILES[self.last_self_tsumo[:2]]] -= 1
                candidates.add(tsumo_tile37*4 + 2 + 0)
                if isMenzenchin:
                    if calc_shanten(convert_vec34_to_short(hand34)) == 0:
                        candidates.add(tsumo_tile37*4 + 2 + 1)
            for idx, pai_num in enumerate(hand37):
                if self.forbidden_tiles[_NUM2TILE[idx]] or pai_num == 0:
                    continue
                candidates.add(idx*4 + 0 + 0)
                hand34_discard = hand34_combine.copy()
                hand34_discard[_TILE37TILE34[idx]] -= 1
                if isMenzenchin:
                    if calc_shanten(convert_vec34_to_short(hand34_discard)) == 0:
                        candidates.add(tsumo_tile37*4 + 0 + 1)
        if self.can_ankan:
            flag = False
            for idx, pai_num in enumerate(self.tehai_vec34):
                if pai_num == 4:
                    candidates.add(148 + idx)
                    flag = True
            if not flag:
                raise RuntimeError(f"can_ankan but didn't found ankan candidates. {self.tehai_vec34}")
        if self.can_kakan:
            flag = False
            for call_event in self.__call_events:
                if call_event["actor"] == self.player_id and call_event["type"] == "pon":
                    idx = VEC34_MJAI_TILES[call_event["pai"][:2]]
                    if self.tehai_vec34[idx] == 1:
                        if idx == 4:
                            if self.akas_in_hand[0]:
                                candidates.add(182 + 0)
                                flag = True
                                continue
                        if idx == 13:
                            if self.akas_in_hand[1]:
                                candidates.add(182 + 10)
                                flag = True
                                continue
                        if idx == 22:
                            if self.akas_in_hand[2]:
                                candidates.add(182 + 20)
                                flag = True
                                continue
                        candidates.add(182 + _TILE34TILE37[idx])
                        flag = True
            if not flag:
                raise RuntimeError(f"can_kakan but didn't found kakan candidates. {self.__call_events} , {self.tehai_vec34}")
        if self.can_tsumo_agari:
            candidates.add(219)
        if self.can_ryukyoku:
            candidates.add(220)
        if self.can_pass:
            candidates.add(221)
        if self.can_chi:
            color = self.last_kawa_tile[1]
            chi_num = int(self.last_kawa_tile[0])
            if (
                self.can_chi_high
                and f"{chi_num-2}{color}" in self.tehai_mjai
                and f"{chi_num-1}{color}" in self.tehai_mjai
            ):
                consumed = [f"{chi_num-2}{color}", f"{chi_num-1}{color}"]
                chi = _CHI2NUM[(self.last_kawa_tile, consumed)]
                candidates.add(222 + chi)
            if (
                self.can_chi_high
                and f"{chi_num-2}{color}r" in self.tehai_mjai
                and f"{chi_num-1}{color}" in self.tehai_mjai
            ):
                consumed = [f"{chi_num-2}{color}r", f"{chi_num-1}{color}"]
                chi = _CHI2NUM[(self.last_kawa_tile, consumed)]
                candidates.add(222 + chi)
            if (
                self.can_chi_high
                and f"{chi_num-2}{color}" in self.tehai_mjai
                and f"{chi_num-1}{color}r" in self.tehai_mjai
            ):
                consumed = [f"{chi_num-2}{color}", f"{chi_num-1}{color}r"]
                chi = _CHI2NUM[(self.last_kawa_tile, consumed)]
                candidates.add(222 + chi)
            if (
                self.can_chi_low
                and f"{chi_num+1}{color}" in self.tehai_mjai
                and f"{chi_num+2}{color}" in self.tehai_mjai
            ):
                consumed = [f"{chi_num+1}{color}", f"{chi_num+2}{color}"]
                chi = _CHI2NUM[(self.last_kawa_tile, consumed)]
                candidates.add(222 + chi)
            if (
                self.can_chi_low
                and f"{chi_num+1}{color}r" in self.tehai_mjai
                and f"{chi_num+2}{color}" in self.tehai_mjai
            ):
                consumed = [f"{chi_num+1}{color}r", f"{chi_num+2}{color}"]
                chi = _CHI2NUM[(self.last_kawa_tile, consumed)]
                candidates.add(222 + chi)
            if (
                self.can_chi_low
                and f"{chi_num+1}{color}" in self.tehai_mjai
                and f"{chi_num+2}{color}r" in self.tehai_mjai
            ):
                consumed = [f"{chi_num+1}{color}", f"{chi_num+2}{color}r"]
                chi = _CHI2NUM[(self.last_kawa_tile, consumed)]
                candidates.add(222 + chi)
            if (
                self.can_chi_mid
                and f"{chi_num-1}{color}" in self.tehai_mjai
                and f"{chi_num+1}{color}" in self.tehai_mjai
            ):
                consumed = [f"{chi_num-1}{color}", f"{chi_num+1}{color}"]
                chi = _CHI2NUM[(self.last_kawa_tile, consumed)]
                candidates.add(222 + chi)
            if (
                self.can_chi_mid
                and f"{chi_num-1}{color}r" in self.tehai_mjai
                and f"{chi_num+1}{color}" in self.tehai_mjai
            ):
                consumed = [f"{chi_num-1}{color}r", f"{chi_num+1}{color}"]
                chi = _CHI2NUM[(self.last_kawa_tile, consumed)]
                candidates.add(222 + chi)
            if (
                self.can_chi_mid
                and f"{chi_num-1}{color}" in self.tehai_mjai
                and f"{chi_num+1}{color}r" in self.tehai_mjai
            ):
                consumed = [f"{chi_num-1}{color}", f"{chi_num+1}{color}r"]
                chi = _CHI2NUM[(self.last_kawa_tile, consumed)]
                candidates.add(222 + chi)
            pass
        if self.can_pon:
            if self.last_kawa_tile[0] == "5" and self.last_kawa_tile[1] != "z":
                if self.tehai_mjai.count(self.last_kawa_tile[:2]) >= 2:
                    consumed = [self.last_kawa_tile[:2], self.last_kawa_tile[:2]]
                    pon = _PENG2NUM[(self.last_kawa_tile, [consumed])]
                    candidates.add(312 + (self.target_actor_rel-1)*40 + pon)
                elif self.tehai_mjai.count(self.last_kawa_tile[:2] + "r") == 1:
                    consumed = [
                        self.last_kawa_tile[:2] + "r",
                        self.last_kawa_tile[:2],
                    ]
                    pon = _PENG2NUM[(self.last_kawa_tile, [consumed])]
                    candidates.add(312 + (self.target_actor_rel-1)*40 + pon)
            else:
                consumed = [
                    self.last_kawa_tile,
                    self.last_kawa_tile,
                ]
                pon = _PENG2NUM[(self.last_kawa_tile, [consumed])]
                candidates.add(312 + (self.target_actor_rel-1)*40 + pon)
            pass
        if self.can_daiminkan:
            candidates.add(432 + (self.target_actor_rel-1)*37 + _TILE2NUM[self.last_kawa_tile])
        if self.can_ron_agari:
            candidates.add(543+self.target_actor_rel-1)
        return list(candidates)


class GameState:
    def __init__(
        self, *, my_name: str, room: int, game_style: int, my_grade: int,
        opponent_grade: int) -> None:
        self.__my_name = my_name
        self.__room = room
        self.__game_style = game_style
        self.__my_grade = my_grade
        self.__opponent_grade = opponent_grade
        self.__seat = None
        self.__player_grades = None
        self.__player_scores = None

    def on_new_game(self) -> None:
        pass

    def on_new_round(self, seat: int, scores: List[int]) -> None:
        self.__seat = seat

        self.__player_grades = [None] * 4
        for i in range(4):
            if i == self.__seat:
                self.__player_grades[i] = self.__my_grade
            else:
                self.__player_grades[i] = self.__opponent_grade

        self.__player_scores = list(scores)

    def __assert_initialized(self) -> None:
        if self.__player_grades is None:
            raise RuntimeError(
                'A method is called on a non-initialized `GameState` object.')
        assert(self.__player_scores is not None)

    def on_liqi_acceptance(self, seat: int) -> None:
        self.__assert_initialized()
        self.__player_scores[seat] -= 1000

    def get_my_name(self) -> str:
        # self.__assert_initialized()
        return self.__my_name

    def get_room(self) -> int:
        self.__assert_initialized()
        return self.__room

    def get_game_style(self) -> int:
        self.__assert_initialized()
        return self.__game_style

    def get_seat(self) -> int:
        self.__assert_initialized()
        return self.__seat

    def get_player_grade(self, seat: int) -> int:
        self.__assert_initialized()
        return self.__player_grades[seat]

    def get_player_rank(self, seat: int) -> int:
        self.__assert_initialized()

        score = self.__player_scores[seat]
        rank = 0
        for i in range(seat):
            if self.__player_scores[i] >= score:
                rank += 1
        for i in range(seat + 1, 4):
            if self.__player_scores[i] > score:
                rank += 1
        assert(0 <= rank and rank < 4)
        return rank

    def get_player_score(self, seat: int) -> int:
        self.__assert_initialized()
        return self.__player_scores[seat]