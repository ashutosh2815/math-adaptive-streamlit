"""
Simple ML engine wrapper.
Loads a trained sklearn model saved with joblib at models/adaptive_tree.pkl
Provides features extraction from Tracker and a predict_action() method.
"""

import os
import numpy as np
import joblib

# default model paths (relative to project root)
_DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "adaptive_tree.pkl")
_DEFAULT_META_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "adaptive_meta.pkl")


class MLEngine:
    def __init__(self, model_path=None, meta_path=None):
        self.model_path = model_path or os.path.abspath(_DEFAULT_MODEL_PATH)
        self.meta_path = meta_path or os.path.abspath(_DEFAULT_META_PATH)
        self.model = None
        self.meta = None
        self._load()

    def _load(self):
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
            else:
                self.model = None
        except Exception as e:
            self.model = None

        try:
            if os.path.exists(self.meta_path):
                self.meta = joblib.load(self.meta_path)
            else:
                self.meta = None
        except Exception:
            self.meta = None

    def features_from_tracker(self, tracker, current_level, window_size=3):
        """
        Produce the model feature vector for the current tracker state:
        [window_accuracy, avg_response_time, streak_correct, level_code]
        Pads with conservative defaults if not enough history.
        """
        window = tracker.last_n(window_size)
        if len(window) < window_size:
            pad_count = window_size - len(window)
            pad = [{'correct': False, 'response_time': 999.0}] * pad_count
            window = pad + window

        n = len(window)
        window_acc = sum(1 for w in window if w['correct']) / n if n > 0 else 0.0
        avg_rt = sum(w['response_time'] for w in window) / n if n > 0 else 999.0
        streak = 0
        for w in reversed(window):
            if w['correct']:
                streak += 1
            else:
                break
        level_map = {'easy': 0, 'medium': 1, 'hard': 2}
        level_code = level_map.get(current_level, 0)
        return np.array([window_acc, avg_rt, streak, level_code]).reshape(1, -1)

    def predict_action(self, tracker, current_level, window_size=3):
        """
        Predict an action using the loaded model.
        Returns: (pred, info) where pred in {-1,0,1} and info may contain 'importance'
        """
        if self.model is None:
            return 0, {"info": "no model loaded"}

        feat = self.features_from_tracker(tracker, current_level, window_size)
        pred = int(self.model.predict(feat)[0])
        importance = None
        try:
            importance = getattr(self.model, "feature_importances_", None)
            if importance is not None:
                importance = importance.tolist()
        except Exception:
            importance = None

        return pred, {"importance": importance}
