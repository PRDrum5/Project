import json
import torch
from collections import OrderedDict
from datetime import datetime

def read_json(file_path):
    with file_path.open('rt') as f:
        return json.load(f, object_hook=OrderedDict)

def fix_seed(seed=0):
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)

class Timer:
    def __init__(self):
        self.cache = datetime.now()

    def check(self):
        now = datetime.now()
        duration = now - self.cache
        self.cache = now
        return duration.total_seconds()

    def reset(self):
        self.cache = datetime.now()
