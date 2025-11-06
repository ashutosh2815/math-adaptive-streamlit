import time
import pandas as pd

class Tracker:
    def __init__(self):
        self.attempts = []  
        self.session_start = None
        self.user = None

    def start_session(self, user_name):
        self.user = user_name
        self.session_start = time.time()
        self.attempts = []

    def record_attempt(self, puzzle, given_answer, correct, response_time):
        entry = {
            'timestamp': time.time(),
            'question_id': puzzle['id'],
            'question': puzzle['question'],
            'level': puzzle['level'],
            'correct': bool(correct),
            'given_answer': given_answer,
            'correct_answer': puzzle['answer'],
            'response_time': float(response_time)
        }
        self.attempts.append(entry)

    def last_n(self, n=3):
        return self.attempts[-n:] if len(self.attempts) >= 1 else []

    def accuracy(self):
        if not self.attempts: return 0.0
        return sum(1 for a in self.attempts if a['correct']) / len(self.attempts)

    def avg_response_time(self):
        if not self.attempts: return 0.0
        return sum(a['response_time'] for a in self.attempts) / len(self.attempts)

    def difficulty_history(self):
        return [a['level'] for a in self.attempts]

    def to_dataframe(self):
        return pd.DataFrame(self.attempts)

    def get_summary(self):
        return {
            'user': self.user,
            'started_at': self.session_start,
            'num_attempts': len(self.attempts),
            'accuracy': self.accuracy(),
            'avg_response_time': self.avg_response_time(),
            'attempts': self.attempts
        }
