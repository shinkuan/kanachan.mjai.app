from mjai import Simulator
import random

submissions = [
    "players/weakml.zip",
    "players/weakml.zip",
    "players/kanachan.zip",
    "players/kanachan.zip",
]
Simulator(submissions, logs_dir="./logs", seed=(random.randint(0, 99999), random.randint(0, 99999)), timeout=10).run()