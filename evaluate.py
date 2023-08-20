import json
import random
from mjai import Simulator
from loguru import logger

class Evaluator:
    def __init__(self, baseline, proposed, n) -> None:
        self.n = n
        self.baseline = baseline
        self.proposed = proposed
        self.cases = [
            [0,0,1,1],
            [0,1,0,1],
            [0,1,1,0],
            [1,0,0,1],
            [1,0,1,0],
            [1,1,0,0]
        ]

        padding = 0
        self.pt = [padding, 90, 45, 0, -135]

        self.result = {}
        self.baseline_rank = 0
        self.proposed_rank = 0
        self.baseline_pt = 0
        self.proposed_pt = 0

    def simulate(self, case, seed):
        submissions = []
        for i in case:
            submissions.append(self.proposed if i else self.baseline)
        Simulator(submissions, logs_dir="./logs", seed=seed, timeout=10).run()
        with open("./logs/summary.json") as f:
            data = json.load(f)
        return data["rank"]

    def rank_pt(self, case, rank):
        for idx, i in enumerate(case):
            if i:
                self.proposed_rank += rank[idx]
                self.proposed_pt += self.pt[rank[idx]]
            else:
                self.baseline_rank += rank[idx]
                self.baseline_pt += self.pt[rank[idx]]

    def run(self):
        for n in range(1, 1+self.n):
            seed = (random.randint(0, 99999), random.randint(0, 99999))
            for case in self.cases:
                rank = self.simulate(case, seed)
                self.rank_pt(case, rank)
            logger.info(f"A set of games are done.")
            logger.info(f"Avg.baseline_rank = {self.baseline_rank / n*6*2}")
            logger.info(f"Avg.proposed_rank = {self.proposed_rank / n*6*2}")
            logger.info(f"baseline_pt = {self.baseline_pt}")
            logger.info(f"proposed_pt = {self.baseline_pt}")
        logger.info(f"All set of games are done.")
        logger.info(f"Avg.baseline_rank = {self.baseline_rank / (self.n*6*2)}")
        logger.info(f"Avg.proposed_rank = {self.proposed_rank / (self.n*6*2)}")
        logger.info(f"baseline_pt = {self.baseline_pt}")
        logger.info(f"proposed_pt = {self.baseline_pt}")
        
Evaluator(baseline="players/kanachan.zip", proposed="players/weakml.zip", n=3).run()