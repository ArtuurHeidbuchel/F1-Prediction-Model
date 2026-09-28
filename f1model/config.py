from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"        
OUTPUT_DIR = ROOT / "output"    
CONFIG_DIR = ROOT / "config"    

API_BASE = "https://api.jolpi.ca/ergast/f1"
FIRST_SEASON = 2018
BACKTEST_START_SEASON = 2024   
TUNE_SEASONS = [2022, 2023]    
MIN_ROUND = 2                  
RACE_DURATION_HOURS = 3      