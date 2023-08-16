#!/usr/bin/env python3

import warnings
from _kanachan import KanachanBot


def main() -> None:
    kanachan = KanachanBot()
    kanachan.start()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    main()