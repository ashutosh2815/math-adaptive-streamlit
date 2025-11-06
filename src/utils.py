import json, time
def timestamp_str(ts=None):
    import datetime
    if ts is None: ts = time.time()
    return datetime.datetime.fromtimestamp(ts).isoformat()

def save_summary_json(summary, path='session_summary.json'):
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)
