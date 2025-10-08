PATH_TO_BEST_MODEL = "models/spine_scan_yolov8x.pt"
DISC_LABELS = ["L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"]
SAGITTAL = [0, 1, 0, 0, 0, -1]
BANNED_TAGS = [
    "ESTIR",
    "STIR",
    "SPAIR",
    "MYELO",
    "NECK",
    "FGRE",
    "DIXON",
    "IR",
    "POSDISP",
    "CUBE",
    "FS",
    "TIRM",
]
