# 🧮 Adaptive Math Learning Prototype

An **AI-powered adaptive math learning system** that dynamically adjusts problem difficulty based on learner performance.  
This hybrid system uses **rule-based heuristics** and a **machine learning model** to personalize learning for children aged 5–10.

---

## 🚀 Features
- Generates math puzzles (Addition, Subtraction, Multiplication, Division)
- Tracks performance: correctness and response time
- Automatically adjusts difficulty (Easy → Medium → Hard)
- Hybrid adaptive logic (Rule-based + ML-based)
- Fallback to rule logic if ML model unavailable
- Displays session summary with accuracy and recommended level

---

## 🧩 Architecture Overview

```
math-adaptive-prototype/
├─ README.md
├─ requirements.txt
├─ src/
│   ├─ main.py
│   ├─ puzzle_generator.py
│   ├─ tracker.py
│   ├─ adaptive_engine.py
│   └─ ml_engine.py
└─ models/
    ├─ adaptive_tree.pkl
    └─ adaptive_meta.pkl
```

**Core Components**
| Module | Description |
|---------|--------------|
| `puzzle_generator.py` | Creates math problems for each difficulty level |
| `tracker.py` | Logs correctness and response times |
| `adaptive_engine.py` | Chooses next difficulty (rule or ML) |
| `ml_engine.py` | Loads and applies ML model for adaptive decisions |
| `main.py` | Orchestrates app flow (CLI / Streamlit) |

---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/math-adaptive-streamlit.git
cd math-adaptive-streamlit
pip install -r requirements.txt
```

---

## ▶️ Running the App

### Option 1 — Command-line
```bash
python src/main.py
```

### Option 2 — Streamlit Interface
```bash
streamlit run app.py
```

---

## 🧠 Adaptive Logic

**Rule-based logic:**
- Promote if accuracy ≥ 0.8 and avg response time ≤ threshold  
- Demote if accuracy ≤ 0.5 or avg response time ≥ threshold  
- Otherwise stay at same level  

**ML-based logic:**
- Decision Tree classifier trained on simulated learner data  
- Features: `[accuracy, avg_response_time, streak, level_code]`  
- Output: `-1` (demote), `0` (stay), `+1` (promote)

**Hybrid approach:**
- Use ML if model available  
- Fall back to rule logic on failure or missing model

---

## 📊 Metrics Tracked
- Accuracy (% correct in window)
- Average response time
- Consecutive correct streak
- Difficulty transition history

---

## 📄 Technical Note
Refer to [`technical_note_adaptive_learning.pdf`](./technical_note_adaptive_learning.pdf)  
for architecture diagrams, design reasoning, and metric definitions.

---

---

## 🧑‍💻 Author
**Ashutosh Shukla**  
AI | ML | Deep Learning | Computer Vision  
# math-adaptive-streamlit
