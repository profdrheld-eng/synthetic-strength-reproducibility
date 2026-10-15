"""Invented test parameters. Not a strength norm or historical estimate."""
import csv
from pathlib import Path

def write_parameters(directory):
    directory=Path(directory)
    with (directory/'coefficients.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['variant','stratum','lift','term','coefficient'])
        for sex in ('female','male'):
            for lift in ('SQ','BP','DL'):
                for term,value in zip(('intercept','height_cm','age_years','age_squared','log_mass_kg','log_mass_squared'),(1.,.002,-.003,0.,.65,0.)):
                    w.writerow(['reduced',sex,lift,term,value])
    with (directory/'residual_scales.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['variant','stratum','lift','sd_residuals'])
        for sex in ('female','male'):
            for lift in ('SQ','BP','DL'):w.writerow(['reduced',sex,lift,.23])
