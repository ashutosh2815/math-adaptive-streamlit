"""
Streamlit app for the Adaptive Math Tutor.
This version integrates ML toggle and handles missing model gracefully.
Run from project root:
    streamlit run src/app.py
"""

import streamlit as st
import time
import json
import pandas as pd

from puzzle_generator import generate_puzzle
from tracker import Tracker

from adaptive_engine import next_level

try:
    from ml_engine import MLEngine
except Exception:
    MLEngine = None

from utils import timestamp_str

st.set_page_config(page_title="Adaptive Math Tutor", layout="wide")

st.sidebar.title("Session controls")

if 'initialized' not in st.session_state:
    st.session_state.initialized = False

name = st.sidebar.text_input("Student name", value=st.session_state.get('user_name', 'Student'))
initial_level = st.sidebar.selectbox("Initial difficulty", options=['easy', 'medium', 'hard'],
                                     index=0 if st.session_state.get('current_level') is None else ['easy','medium','hard'].index(st.session_state.get('current_level')))
rounds = st.sidebar.number_input("Planned rounds", min_value=3, max_value=100, value=12, step=1)
window_size = st.sidebar.slider("Adaptive window (N)", min_value=1, max_value=6, value=3)
enable_ml = st.sidebar.checkbox("Enable ML engine", value=False)
consent_save = st.sidebar.checkbox("Consent to save anonymized session data", value=False)

start_btn = st.sidebar.button("Start session")
end_btn = st.sidebar.button("End session and export")

if enable_ml:
    st.sidebar.markdown("### ML model info")
    if MLEngine is not None:
        try:
            me = MLEngine()
            if me.model is None:
                st.sidebar.write("No model found. Run `python src/train_model.py` to create models/adaptive_tree.pkl")
            else:
                imps = getattr(me.model, "feature_importances_", None)
                if imps is not None:
                    feat_names = ['window_acc', 'avg_rt', 'streak', 'level_code']
                    for n, v in zip(feat_names, imps):
                        st.sidebar.write(f"{n}: {v:.3f}")
                else:
                    st.sidebar.write(f"Loaded model: {type(me.model).__name__}")
        except Exception as e:
            st.sidebar.write("Error loading model:", e)
    else:
        st.sidebar.write("ML engine not available (ml_engine import failed)")

if start_btn:
    st.session_state.tracker = Tracker()
    st.session_state.tracker.start_session(name)
    st.session_state.current_level = initial_level
    st.session_state.rounds_left = int(rounds)
    st.session_state.initialized = True
    st.session_state.history = []
    st.session_state.awaiting_answer = False
    st.session_state.user_name = name
    st.experimental_rerun()

if not st.session_state.initialized:
    st.title("Adaptive Math Tutor — Streamlit")
    st.markdown("Set name and press **Start session** in the sidebar to begin.")
    st.stop()

col1, col2 = st.columns([2, 1])

with col1:
    st.header("Problem")
    if not st.session_state.get('awaiting_answer', False):
        puzzle = generate_puzzle(level=st.session_state.current_level)
        st.session_state.last_puzzle = puzzle
        st.session_state.awaiting_answer = True
        st.session_state.answer_start = time.time()

    puzzle = st.session_state.last_puzzle
    st.markdown(f"**Level:** `{puzzle['level'].upper()}`")
    st.markdown(f"### {puzzle['question']}")
    user_answer = st.text_input("Your answer", key="answer_input")
    submit = st.button("Submit Answer")

    if submit:
        rt = time.time() - st.session_state.answer_start
        given = user_answer.strip()
        correct = False
        try:
            correct = abs(float(given) - float(puzzle['answer'])) < 1e-6
        except Exception:
            correct = (given == str(puzzle['answer']))

        st.session_state.tracker.record_attempt(puzzle, given, correct, rt)
        st.session_state.history.append({
            'question': puzzle['question'],
            'level': puzzle['level'],
            'given': given,
            'correct': correct,
            'rt': rt
        })

        next_lvl, reason = next_level(st.session_state.tracker,
                                      st.session_state.current_level,
                                      window_size=window_size,
                                      use_ml=enable_ml)
        prev_lvl = st.session_state.current_level
        st.session_state.current_level = next_lvl

        if correct:
            st.success(f"Correct! ✅ (response time: {rt:.1f}s)")
        else:
            st.error(f"Incorrect. Correct answer: **{puzzle['answer']}** (response time: {rt:.1f}s)")

        st.info(f"Adaptive decision: **{next_lvl.upper()}** — {reason} (from {prev_lvl.upper()})")

        st.session_state.rounds_left -= 1
        st.session_state.awaiting_answer = False
        st.session_state.answer_input = ""
        st.experimental_rerun()

with col2:
    st.header("Session summary")
    tracker = st.session_state.tracker
    st.metric("Attempts", len(tracker.attempts))
    st.metric("Accuracy", f"{tracker.accuracy()*100:.1f}%")
    st.metric("Avg response (s)", f"{tracker.avg_response_time():.1f}")

    st.markdown("**Last attempts**")
    if tracker.attempts:
        df = pd.DataFrame(tracker.attempts)
        st.dataframe(df.tail(8)[['timestamp', 'question', 'level', 'correct', 'response_time']])
    else:
        st.write("No attempts recorded yet.")

    st.markdown("---")
    st.write("Current level:", st.session_state.current_level.upper())
    st.write("Rounds left:", st.session_state.rounds_left)
    hist = [a['level'] for a in tracker.attempts]
    if hist:
        counts = pd.Series(hist).value_counts().reindex(['easy', 'medium', 'hard']).fillna(0)
        st.bar_chart(counts)

# End session behavior
if end_btn or (st.session_state.rounds_left <= 0):
    summary = tracker.get_summary()
    summary['ended_at'] = time.time()
    df = tracker.to_dataframe()
    csv = df.to_csv(index=False).encode('utf-8')

    st.download_button("Download attempts CSV", data=csv,
                       file_name=f"session_{tracker.user}_{timestamp_str()}.csv")
    st.download_button("Download summary JSON",
                       data=json.dumps(summary, indent=2),
                       file_name=f"session_summary_{tracker.user}.json")

    if consent_save:
        try:
            import os
            os.makedirs("data", exist_ok=True)
            csvpath = f"data/session_{tracker.user}_{timestamp_str()}.csv"
            df.to_csv(csvpath, index=False)
            st.write("Saved anonymized session data to", csvpath)
        except Exception as e:
            st.write("Error saving session data:", e)

    st.success("Session ended. You may restart or close the app.")
    st.session_state.initialized = False
    st.session_state.awaiting_answer = False
    st.stop()
