import numpy as np

PHI_MIN = np.pi/2 - np.pi/12
PHI_MAX = np.pi/2 + np.pi/12
R_MIN = 800.0
R_MAX = 5000.0
ETA_MIN = 1.22
ETA_MAX = 2.8

PADS = 1739
LAYERS = 8
QUADS = 3

PADS_PER_LAYER = {}
PADS_PER_LAYER[0] = list(range(0, 218))
PADS_PER_LAYER[1] = list(range(218, 430))
PADS_PER_LAYER[2] = list(range(430, 671))
PADS_PER_LAYER[3] = list(range(671, 920))
PADS_PER_LAYER[4] = list(range(920, 1124))
PADS_PER_LAYER[5] = list(range(1124, 1328))
PADS_PER_LAYER[6] = list(range(1328, 1531))
PADS_PER_LAYER[7] = list(range(1531, 1739))

PADS_REQUIRED = 8

ATLAS_ETA_MU = 0.0
ATLAS_ETA_SIGMA = 2.0

ZS = [
    7454.0,
    7465.0,
    7476.0,
    7487.0,
    7787.0,
    7798.0,
    7809.0,
    7820.0,
]
ZMID = sum(ZS) / len(ZS)

RMAXS = [
    2260.0,
    3470.0,
    # 4650.0,
    5200.0,
]

SECTOR_XMIN = -1030.0
SECTOR_XMAX = 1030.0
SECTOR_YMIN = 940.0
SECTOR_YMAX = 4620.0
