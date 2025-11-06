"""
Adaptive engine with both rule-based and ML-backed decision.
This module tries to import MLEngine robustly (ml_engine or src.ml_engine).
If ML engine cannot be imported or model file missing, it falls back to rules.
"""

try:
    from ml_engine import MLEngine
except Exception:
    try:
        from src.ml_engine import MLEngine
    except Exception:
        MLEngine = None

LEVELS = ['easy', 'medium', 'hard']

FAST_THRESH = {'easy': 8.0, 'medium': 12.0, 'hard': 20.0}
SLOW_THRESH = {'easy': 15.0, 'medium': 20.0, 'hard': 30.0}


def increase(level):
    idx = LEVELS.index(level)
    return LEVELS[min(idx + 1, len(LEVELS) - 1)]


def decrease(level):
    idx = LEVELS.index(level)
    return LEVELS[max(idx - 1, 0)]


def next_level_rule(tracker, current_level, window_size=3):
    """
    Rule-based adaptive decision:
    - Promote if window_accuracy >= 0.8 and avg_response_time <= FAST_THRESH
    - Demote if window_accuracy <= 0.5 or avg_response_time >= SLOW_THRESH
    - Otherwise stay
    Returns: (next_level, reason_str)
    """
    window = tracker.last_n(window_size)
    if not window:
        return current_level, "no data yet (rule)"

    acc = sum(1 for a in window if a['correct']) / len(window)
    avg_time = sum(a['response_time'] for a in window) / len(window)

    if acc >= 0.8 and avg_time <= FAST_THRESH[current_level]:
        return increase(current_level), f"Promote (rule): acc={acc:.2f}, time={avg_time:.1f}s"
    if acc <= 0.5 or avg_time >= SLOW_THRESH[current_level]:
        return decrease(current_level), f"Demote (rule): acc={acc:.2f}, time={avg_time:.1f}s"
    return current_level, f"Stay (rule): acc={acc:.2f}, time={avg_time:.1f}s"


# Global ML engine instance (lazy-loaded)
_ml_engine = None


def next_level(tracker, current_level, window_size=3, use_ml=False):
    """
    Unified API to get next difficulty level.
    If use_ml is True and a model is available, use ML prediction.
    Fall back to rule-based decision if model unavailable or on error.
    Returns: (next_level, reason)
    """
    global _ml_engine

    if use_ml and MLEngine is not None:
        if _ml_engine is None:
            try:
                _ml_engine = MLEngine()
            except Exception as e:
                _ml_engine = None

        if _ml_engine and getattr(_ml_engine, "model", None) is not None:
            try:
                pred, info = _ml_engine.predict_action(tracker, current_level, window_size=window_size)
                if pred == 1:
                    return increase(current_level), f"Promote (ML) — importance={info.get('importance')}"
                if pred == -1:
                    return decrease(current_level), f"Demote (ML) — importance={info.get('importance')}"
                return current_level, f"Stay (ML) — importance={info.get('importance')}"
            except Exception as e:
                return next_level_rule(tracker, current_level, window_size)
        else:
            return next_level_rule(tracker, current_level, window_size)
    else:
        return next_level_rule(tracker, current_level, window_size)
