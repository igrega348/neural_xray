from pathlib import Path
import pandas as pd
import cv2 as cv
import numpy as np
from typing import Optional
from rich.progress import track
from datetime import datetime

class PrintTableMetrics:
    def __init__(self, log_metrics: list, col_width: int = 12, max_iter: Optional[int] = None) -> None:
        super().__init__()

        header = []
        for metric in log_metrics:
            header.append(metric)
        if 'Iteration' not in header:
            header.insert(0, "Iteration")
        if 'Time' not in header:
            header.insert(0, "Time")
        if max_iter is not None:
            header.append('ETA')
        if 'it/s' not in header:
            header.append('it/s')
        
        self.format_str = '{' + ':>' + str(col_width) + '}'
        self.col_width = col_width
        n_cols = len(header)
        total_width = col_width * n_cols + 3*n_cols
        self.total_width = total_width
        self.header = header
        self._time_metrics = {'start':datetime.now(), 'max_iter':max_iter}
        fields = [self.format_str.format(metric) for metric in self.header]
        line = " | ".join(fields) + "\n" + "-" * self.total_width
        print(line)
    
    def update(self, metrics: dict) -> str:
        # Formatting
        s = self.format_str
        if 'Time' not in metrics:
            metrics['Time'] = datetime.now().strftime('%H:%M:%S')

        cur_iter = max(1, metrics['Iteration'])
        time_per_iter = (datetime.now() - self._time_metrics['start']) / cur_iter
        iter_per_s = 1 / time_per_iter.total_seconds()
        metrics['it/s'] = iter_per_s

        if 'ETA' in self.header:
            assert 'Iteration' in metrics
            assert self._time_metrics['max_iter'] is not None
            iter_to_go = self._time_metrics['max_iter'] - cur_iter
            time_left = time_per_iter * iter_to_go
            seconds_left = time_left.total_seconds()
            metrics['ETA'] = f'{seconds_left//3600:.0f}H{(seconds_left%3600)//60:.0f}m{seconds_left%60:.0f}s'

        fields = []
        for key in self.header:
            if key in metrics:
                if isinstance(metrics[key], float):
                    val = f'{metrics[key]:.6f}'
                else:
                    val = metrics[key]
                fields.append(s.format(val))
            else:
                fields.append(s.format(''))
        line =  " | ".join(fields)
        print(line)