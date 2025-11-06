from src.puzzle_generator import generate_puzzle
from src.tracker import Tracker
from src.adaptive_engine import next_level_rule

def test_generate():
    p = generate_puzzle('easy', seed=1)
    assert 'question' in p and 'answer' in p

def test_tracker_and_engine():
    t = Tracker(); t.start_session("t")
    p = generate_puzzle('easy', seed=2)
    t.record_attempt(p, given_answer=str(p['answer']), correct=True, response_time=5.0)
    next_level, reason = next_level_rule(t, 'easy', window_size=1)
    assert next_level in ['easy','medium']
