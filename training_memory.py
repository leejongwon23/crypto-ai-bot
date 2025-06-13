from failure_db import load_existing_failure_hashes
from wrong_data_loader import load_training_prediction_data
from collections import Counter
import sqlite3

def get_frequent_failures(min_count=5):
    counter = Counter()
    try:
        with sqlite3.connect("/persistent/logs/failure_patterns.db") as conn:
            rows = conn.execute("SELECT hash FROM failure_patterns").fetchall()
            for row in rows:
                counter[row[0]] += 1
    except:
        return set()
    return {h for h, cnt in counter.items() if cnt >= min_count}


def load_training_prediction_data(symbol, strategy, input_size, window, source_type="wrong"):
    return load_training_prediction_data(symbol, strategy, input_size, window, source_type)
