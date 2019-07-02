import logging
import logging.config
from pathlib import Path
from utils import read_json

def configure_logging(log_dir, config_dir='logger/logging_config.json', 
                      default_level=logging.INFO):
    config_dir = Path(config_dir)
    log_config = read_json(config_dir)
    log_config['handlers']['info_file_handler']['filename'] = str(log_dir / log_config['handlers']['info_file_handler']['filename']) 

    logging.config.dictConfig(log_config)