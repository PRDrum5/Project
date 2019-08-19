import os
import logging
from utils import read_json
from pathlib import Path
from datetime import datetime
from logger import configure_logging

class GetConfig:
    def __init__(self, config_file):
        self.config_file = Path(config_file)

        self.config = read_json(self.config_file)

        train_model = self.config['train_mode']

        if train_model:
            experiment_name = self.config['experiment_name']
            out_dir = Path(self.config['trainer']['save_dir'])
            timestamp = datetime.now().strftime(r'%Y%m%d_%H%M%S')

            self.save_dir = out_dir / 'models' / experiment_name / timestamp
            self.log_dir = out_dir / 'log' / experiment_name / timestamp
            self.train_samples_dir = out_dir / 'samples' / 'train' / experiment_name / timestamp
            self.test_samples_dir = out_dir / 'samples' / 'test' / experiment_name / timestamp

            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.train_samples_dir.mkdir(parents=True, exist_ok=True)
            self.test_samples_dir.mkdir(parents=True, exist_ok=True)

            configure_logging(self.log_dir)
            self.log_levels = {
                0: logging.WARNING,
                1: logging.INFO,
                2: logging.DEBUG,
            }
        
    def get(self, name, module, *args, **kwargs):
        if ',' in name:
            model, name = name.split(',')
            module_name = self[model][name]['spec']
            module_args = dict(self[model][name]['args'])
        else:
            module_name = self[name]['spec']
            module_args = dict(self[name]['args'])
        collected_object = getattr(module, module_name)(*args, **module_args)
        return collected_object

    def get_logger(self, name, level=2):
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[level])
        return logger
    
    def get_func(self, name, module):
        if ',' in name:
            model, name = name.split(',')
            module_name = self[model][name]['spec']
        else:
            module_name = self[name]['spec']
        return getattr(module, module_name)

    def __getitem__(self, name):
        return self.config[name]