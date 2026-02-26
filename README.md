# 🎓 Student Marks Analysis & Prediction Dashboard

A Machine Learning-powered web application that analyses student academic performance and predicts final exam marks using Linear Regression. Built with Python and Streamlit, the dashboard provides interactive visualisations, individual student lookups, and model transparency — all wrapped in a sleek dark-themed UI.

---

## 📸 Features at a Glance

| Tab | What it shows |
|---|---|
| 📋 **Overview** | Full student table with descriptive statistics |
| 📊 **Charts** | Class averages, score distributions, attendance breakdown |
| 🎯 **Predictions** | Predicted final marks, grades (O → F), pass/fail status |
| 🔍 **Student Lookup** | Individual student vs class average comparison |
| 🤖 **Model Info** | MAE, R² score, feature coefficients, actual vs predicted scatter |

---

## 🧠 How It Works

The app takes **two CSV files** as input:

1. **Training Data** — Historical student records with actual final marks. Used to train the Linear Regression model.
2. **Student Marks Data** — Current students whose final marks need to be predicted.

The model learns the relationship between four features and the final score, then predicts finals for each current student.

```
Predicted Finals = w₁×Attendance + w₂×CIA-1 + w₃×Mid-Sem + w₄×CIA-3 + intercept
```

---

## 📁 Project Structure

```
├── student_app.py              # Main Streamlit application
├── training_data.csv           # Historical student data (with finals column)
├── students_marks_data.csv     # Current student data (for prediction)
└── README.md
```

---

## 📊 CSV Format

### `training_data.csv` — for model training
| Column | Description | Scale |
|---|---|---|
| `Register No` | Student registration number | — |
| `Name` | Student name | — |
| `Attendence` | Attendance percentage | 0–100% |
| `CIA-1` | Continuous Internal Assessment 1 | Out of 20 |
| `Mid-sem` | Mid-semester exam | Out of 50 |
| `CIA-3` | Continuous Internal Assessment 3 | Out of 20 |
| `finals` | **Actual** final exam marks *(required for training)* | Out of 100 |

### `students_marks_data.csv` — for prediction
Same columns as above, **without** the `finals` column.

---

## 🤖 Machine Learning Details

**Algorithm:** Linear Regression (`sklearn.linear_model.LinearRegression`)

**Why Linear Regression?**
The relationship between internal assessments and final performance is expected to be roughly linear — students who score higher in CIA and Mid-Sem consistently tend to perform better in finals. Linear Regression is interpretable, fast, and effective for this kind of academic prediction.

**Feature Engineering:**
All features are normalised to a 0–100 scale before training to prevent higher-range features (like Mid-Sem out of 50) from dominating lower-range ones (like CIA-1 out of 20).

```python
cia1_sc = (CIA-1 / 20) × 100
mid_sc  = (Mid-sem / 50) × 100
cia3_sc = (CIA-3 / 20) × 100
att_sc  = Attendance (already in %)
```

**Train-Test Split:** 80% training / 20% testing (`random_state=42`)

**Evaluation Metrics:**
- **MAE** (Mean Absolute Error) — average mark difference between predicted and actual
- **R² Score** — how well the model explains variance in final marks (closer to 1.0 is better)

---

## 📈 Visualisations

- **Class Average Bar Chart** — Average vs max marks per exam
- **Score Distribution Histograms** — Mark spread across CIA-1, Mid-Sem, CIA-3
- **Attendance Donut Chart** — Students categorised by attendance bracket (<60%, 60–75%, 75–90%, >90%)
- **Attendance vs Predicted Finals Scatter** — Trend line showing correlation
- **Student vs Class Average** — Individual comparison bar chart
- **Predicted Grade Distribution** — Count of O, A+, A, B+, B, C, F grades
- **Actual vs Predicted Scatter** — Model accuracy visualisation

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `streamlit` | Web application framework |
| `pandas` | Data loading, cleaning, and manipulation |
| `numpy` | Numerical operations and array handling |
| `scikit-learn` | Machine Learning — Linear Regression, train-test split, metrics |
| `matplotlib` | Chart creation and styling |
| `seaborn` | Distribution histograms |
| `io`, `os`, `warnings` | File handling and environment utilities |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/student-marks-dashboard.git
cd student-marks-dashboard
```

### 2. Install dependencies
```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn
```

### 3. Run the app
```bash
streamlit run student_app.py
```

### 4. Upload your files
Use the **sidebar** to upload:
- Your `training_data.csv`
- Your `students_marks_data.csv`

> **Note:** If the default CSV files (`training_data.csv` and `students marks data.csv`) are present in the same directory as `student_app.py`, they will be loaded automatically without needing to upload.

---

## 🎓 Grading Scale

| Grade | Marks Range |
|---|---|
| O | ≥ 90 |
| A+ | 80 – 89 |
| A | 70 – 79 |
| B+ | 60 – 69 |
| B | 50 – 59 |
| C | 40 – 49 |
| F | < 40 |

Students with predicted finals ≥ 40 are marked **Pass ✅**, otherwise **Fail ❌**.

---

## 📌 Notes

- Attendance values are accepted as both `78.40%` (string) and `78.40` (float).
- Predicted marks are clipped between 0 and 100 to ensure realistic outputs.
- The model is cached using `@st.cache_resource` so it only retrains when the training file changes.

---

## 👨‍💻 Author

Built as an academic ML project demonstrating student performance analysis and prediction using supervised learning.