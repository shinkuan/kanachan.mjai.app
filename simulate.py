from mjaisimulator import Simulator

submissions = [
    "players/kanachan.zip",
    "players/tsumogiri.zip",
    "players/tsumogiri.zip",
    "players/invalidbot2.zip",
]
Simulator(submissions, logs_dir="./logs").run()