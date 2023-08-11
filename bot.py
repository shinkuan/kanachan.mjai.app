#!/usr/bin/env python3

import warnings
from _kanachan import Kanachan


def main() -> None:
    kanachan = Kanachan()
    kanachan.run()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    main()