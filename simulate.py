from mjai import Simulator
import random

submissions = [
    "players/weakml.zip",
    "players/rulebase.zip",
    "players/rulebase.zip",
    "players/rulebase.zip",
]
Simulator(submissions, logs_dir="./logs", seed=(random.randint(0, 99999), random.randint(0, 99999))).run()