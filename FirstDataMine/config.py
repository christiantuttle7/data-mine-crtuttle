# config.py
import os
DATA_DIR = "data"
LOCAL_TZ = "America/Denver"
DEFAULT_LOCATIONS = {
    "Grand Junction, CO": (39.0639, -108.5506),
    "Fruita, CO": (39.1589, -108.7280),
    "Palisade, CO": (39.1108, -108.3509),
    "Montrose, CO": (38.4783, -107.8762),
}
os.makedirs(DATA_DIR, exist_ok=True)
