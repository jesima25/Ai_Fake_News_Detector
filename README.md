# 🛡️ FakeNewsIQ — AI-Powered Fake News Verification Platform

> Detect fake news instantly with Explainable AI, Multi-language support, Video analysis, and Source credibility checking.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-black?style=flat-square&logo=flask)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=flat-square&logo=scikit-learn)
![Accuracy](https://img.shields.io/badge/Accuracy-99%25-brightgreen?style=flat-square)

---

## 📌 About

**FakeNewsIQ** is a machine learning web application that detects whether a news article is **FAKE or REAL**. Unlike basic detectors, it explains *why* the prediction was made, supports multiple Indian languages, and can analyze YouTube videos and uploaded video files.

- 👩‍💻 Developer: **Jesima**
- 🎓 Final Year Project — 2026
- 🏫 Department of Computer Science


## ✨ Features

| Feature | Description |
|---|---|
| 🤖 ML Prediction | Logistic Regression trained on 44,000+ articles — 99% accuracy |
| 🧠 Explainable AI | Shows exactly WHY news is FAKE or REAL with signals |
| 📊 Linguistic Metrics | Sensationalism, emotional language, factual references, clickbait score |
| 🌍 Multi-Language | Tamil, Hindi, Telugu, Bengali, French, Arabic support |
| 🎦 Video Analysis | Upload video file → extract speech → analyze |
| 📺 YouTube Analysis | Paste YouTube URL → extract captions → analyze |
| 🌐 Source Credibility | Domain trust score checker (BBC=97, Infowars=5) |
| 🔊 Text to Speech | Reads full result, reason and signals aloud |
| 📈 Dashboard | Fake vs Real bar chart with prediction history |
| 💾 Persistent History | SQLite database — predictions saved across sessions |


## 🖥️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML, CSS, JavaScript |
| Backend | Python, Flask, Flask-CORS |
| ML Model | Scikit-learn — Logistic Regression |
| Text Features | TF-IDF Vectorizer (50,000 features) |
| Translation | googletrans |
| Video Processing | moviepy, SpeechRecognition, pydub |
| YouTube | youtube-transcript-api |
| Database | SQLite |
| Charts | Chart.js |
| Speech | Web Speech API |

---

## 📁 Project Structure

```
Ai_Fake_News_Detector/
│
├── api.py                  # Flask backend — all routes and ML logic
├── train_model.py          # Train and save the ML model
├── check_data.py           # Verify dataset before training
│
├── index.html              # Home page — analyze news
├── history.html            # Prediction history page
│
├── fake_news_model.pkl     # Trained Logistic Regression model
├── vectorizer.pkl          # Fitted TF-IDF vectorizer
│
├── predictions.db          # SQLite history database
├── requirements.txt        # Python dependencies
└── Procfile                # Deployment config
```


## 🧠 How It Works

```
User Input (Text / Video / YouTube URL)
            ↓
   Language Detection
            ↓
   Auto Translation (if non-English)
            ↓
   Text Cleaning (lowercase, remove URLs)
            ↓
   TF-IDF Vectorization (50,000 features)
            ↓
   Logistic Regression Prediction
            ↓
   NLP Signal Detection (why FAKE or REAL)
            ↓
   Linguistic Metrics Calculation
            ↓
   Source Credibility Check
            ↓
   Result → FAKE or REAL + Full Explanation
```

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Accuracy | 99%+ |
| F1 Score | 99.5% |
| Training Articles | 44,000+ |
| Algorithm | Logistic Regression |
| TF-IDF Features | 50,000 words |
| Train / Test Split | 80% / 20% |

---

## 🌍 Supported Languages

Tamil · Hindi  · English

---

## 💡 Sample Test

**🔴 Fake News Input:**
```
SHOCKING secret doctors will never tell you about this miracle cure
that big pharma is hiding. Anonymous sources leaked documents proving
government suppresses natural remedies. WAKE UP patriots and share
this before the deep state deletes it forever tonight!
```
**Result → FAKE (94% confidence)**

**🟢 Real News Input:**
```
The Reserve Bank of India raised its benchmark interest rate by 25
basis points on Wednesday according to official statement. The decision
was unanimous among the six member monetary policy committee. Governor
confirmed policymakers want further evidence of cooling inflation.
```
**Result → REAL (97% confidence)**


## 📄 License

This project is licensed under the MIT License.

⭐ **If you found this project helpful, please give it a star!**
