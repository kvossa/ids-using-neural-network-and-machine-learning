import logging
import sys
from pathlib import Path
import json
from datetime import datetime

class IDSLogger:
    def __init__(self, name='ids', log_dir='logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        self.logger.handlers = []
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_format)
        
        log_file = self.log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    def log(self, level, message, **kwargs):
        """Structured logging with additional context"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message,
            **kwargs
        }
        self.logger.debug(json.dumps(log_entry))
        
        if level.upper() == 'INFO':
            self.logger.info(message)
        elif level.upper() == 'WARNING':
            self.logger.warning(message)
        elif level.upper() == 'ERROR':
            self.logger.error(message)
    
    def log_preprocessing(self, df_before, df_after, operation):
        """Specialized log for preprocessing steps"""
        stats = {
            'operation': operation,
            'rows_before': len(df_before),
            'rows_after': len(df_after),
            'columns_before': list(df_before.columns),
            'columns_after': list(df_after.columns),
            'removed_columns': set(df_before.columns) - set(df_after.columns)
        }
        self.log('INFO', f"Preprocessing: {operation}", stats=stats)

if __name__ == "__main__":
    logger = IDSLogger()
    logger.log('INFO', 'Starting IDS pipeline', stage='initialization')
    
    import pandas as pd
    df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    logger.log_preprocessing(df, df.drop(columns=['col2']), 'drop_columns')