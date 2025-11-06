import random
import operator

OPS = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '/': lambda a, b: a // b if b != 0 and a % b == 0 else round(a / b, 2)
}

LEVEL_CONFIG = {
    'easy': {
        'ops': ['+','-'],
        'range': (1, 10)
    },
    'medium': {
        'ops': ['+','-','*'],
        'range': (5, 30)
    },
    'hard': {
        'ops': ['+','-','*','/'],
        'range': (10, 150)
    }
}

def generate_puzzle(level='easy', seed=None):
    """Return dict: {id, question, answer, level, metadata}"""
    if seed is not None:
        random.seed(seed)
    conf = LEVEL_CONFIG.get(level, LEVEL_CONFIG['easy'])
    a = random.randint(*conf['range'])
    b = random.randint(*conf['range'])
    op = random.choice(conf['ops'])
    # prevent divide-by-zero
    if op == '/' and b == 0: b = 1
    # For easy, avoid negative results (optional)
    if level == 'easy' and op == '-' and a < b:
        a, b = b, a
    answer = OPS[op](a, b)
    qtext = f"{a} {op} {b} = ?"
    return {
        'id': f"{level}_{random.getrandbits(32)}",
        'question': qtext,
        'answer': answer,
        'level': level,
        'op': op,
        'operands': (a, b)
    }
