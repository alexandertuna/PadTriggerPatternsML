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
    5000.0,
]

SECTOR_XMIN = -1030.0
SECTOR_XMAX = 1030.0
SECTOR_YMIN = 940.0
SECTOR_YMAX = 4620.0
