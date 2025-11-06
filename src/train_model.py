"""
Train a small adaptive decision model using simulated user sessions.
Saves model to models/adaptive_tree.pkl and a small metadata file.
"""
import os
import random
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Simple feature extractor from a sliding window of attempts
def make_features_from_window(window, current_level):

    n = len(window)
    window_accuracy = sum(1 for w in window if w['correct']) / n if n>0 else 0.0
    avg_rt = sum(w['response_time'] for w in window) / n if n>0 else 999.0
    streak = 0
    for w in reversed(window):
        if w['correct']:
            streak += 1
        else:
            break
    level_map = {'easy':0, 'medium':1, 'hard':2}
    level_code = level_map.get(current_level, 0)
    return [window_accuracy, avg_rt, streak, level_code]

def simulate_session(skill_level='medium', session_length=20):

    profile = {
        'low':  (0.4, 15.0),   
        'medium':(0.7, 10.0),
        'high': (0.9, 6.0)
    }[skill_level]
    p_correct, avg_rt = profile
    attempts = []
    for i in range(session_length):
        correct = random.random() < p_correct
        rt = max(0.5, random.gauss(avg_rt, avg_rt*0.3))
        attempts.append({'correct': correct, 'response_time': rt})
    return attempts

def label_action_from_window(window, current_level):
    acc = sum(1 for w in window if w['correct']) / len(window)
    avg_rt = sum(w['response_time'] for w in window) / len(window)
    FAST = {'easy':8.0,'medium':12.0,'hard':20.0}
    SLOW = {'easy':15.0,'medium':20.0,'hard':30.0}
    if acc >= 0.8 and avg_rt <= FAST[current_level]:
        return 1
    if acc <= 0.5 or avg_rt >= SLOW[current_level]:
        return -1
    return 0

def generate_dataset(num_sessions=2000, window_size=3):
    X = []
    y = []
    levels = ['easy','medium','hard']
    skills = ['low','medium','high']
    for _ in range(num_sessions):
        skill = random.choice(skills)
        current_level = random.choice(levels)
        attempts = simulate_session(skill, session_length=20)
        for i in range(window_size, len(attempts)+1):
            window = attempts[i-window_size:i]
            feat = make_features_from_window(window, current_level)
            act = label_action_from_window(window, current_level)
            X.append(feat)
            y.append(act)
    return pd.DataFrame(X, columns=['window_acc','avg_rt','streak','level_code']), pd.Series(y, name='action')

def main():
    os.makedirs('models', exist_ok=True)
    print("Generating dataset...")
    X, y = generate_dataset(num_sessions=2500, window_size=3)
    print("Dataset shape:", X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Training DecisionTreeClassifier...")
    clf = DecisionTreeClassifier(max_depth=6, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification report:\n", classification_report(y_test, y_pred))
    joblib.dump(clf, "models/adaptive_tree.pkl")
    meta = {
        'model': 'DecisionTreeClassifier',
        'window_size': 3,
        'features': ['window_acc','avg_rt','streak','level_code']
    }
    joblib.dump(meta, "models/adaptive_meta.pkl")
    print("Saved model to models/adaptive_tree.pkl and models/adaptive_meta.pkl")
    X_test.assign(action=y_test).to_csv("models/test_examples.csv", index=False)

if __name__ == "__main__":
    main()
