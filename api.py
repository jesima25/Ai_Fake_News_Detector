# =====================================
# api.py — FakeNewsIQ v3.0
# Author: Jesima
# =====================================

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib, re, numpy as np, os, sqlite3, datetime

app = Flask(__name__)
CORS(app)

FOLDER  = r"C:\Temp\FakeNewsProject"
DB_PATH = os.path.join(FOLDER, "predictions.db")

model = joblib.load(os.path.join(FOLDER, "models", "model.pkl"))
tfidf = joblib.load(os.path.join(FOLDER, "models", "tfidf.pkl"))

# ── Database ──────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    # Create table if not exists
    conn.execute("""CREATE TABLE IF NOT EXISTS predictions
        (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT,
         verdict TEXT, conf REAL, created TEXT)""")
    # Add new columns safely (won't fail if already exist)
    for col in ["lang TEXT DEFAULT 'en'", "source TEXT DEFAULT 'text'"]:
        try:
            conn.execute(f"ALTER TABLE predictions ADD COLUMN {col}")
        except:
            pass
    conn.commit(); conn.close()
init_db()

# ── Helpers ───────────────────────────────────
def clean(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def get_keywords(text):
    vec    = tfidf.transform([text])
    words  = tfidf.get_feature_names_out()
    scores = vec.toarray()[0]
    top    = np.argsort(scores)[::-1][:6]
    return [words[i] for i in top if scores[i] > 0]

def translate(text, lang):
    if lang == "en": return None
    try:
        from googletrans import Translator
        return Translator().translate(text, src=lang, dest="en").text
    except:
        return None

# ── Signal detection (why FAKE or REAL) ───────
FAKE_WORDS  = ["shocking","secret","exposed","leaked","deep state","big pharma",
               "wake up","corrupt","globalist","suppressed","anonymous","bombshell"]
REAL_WORDS  = ["reuters","according to","confirmed","official","published","university",
               "government","study","percent","announced","bbc","ndtv"]
EMO_WORDS   = ["furious","outraged","terrifying","evil","patriots","destroy","disgusting"]
CLICK_WORDS = ["share now","before it's deleted","you won't believe","tell everyone","forward this"]

def get_signals(text, verdict):
    t  = text.lower()
    signals = []

    found_fake = [w for w in FAKE_WORDS if w in t]
    if found_fake:
        signals.append({"icon":"⚠️","type":"danger","text":f"Sensational words: {', '.join(found_fake[:3])}"})

    found_emo = [w for w in EMO_WORDS if w in t]
    if found_emo:
        signals.append({"icon":"😤","type":"warn","text":f"Emotional language: {', '.join(found_emo[:3])}"})

    if any(w in t for w in CLICK_WORDS) or text.count("!") > 3:
        signals.append({"icon":"📢","type":"warn","text":"Clickbait or urgency language detected"})

    found_real = [w for w in REAL_WORDS if w in t]
    if found_real:
        signals.append({"icon":"🏛️","type":"ok","text":f"Credible references: {', '.join(found_real[:3])}"})
    elif verdict == "FAKE":
        signals.append({"icon":"📚","type":"danger","text":"No credible source attribution found"})

    caps = [w for w in text.split() if w.isupper() and len(w) > 2]
    if len(caps) > 2:
        signals.append({"icon":"🔠","type":"warn","text":f"Excessive caps: {' '.join(caps[:3])}"})

    if verdict == "REAL":
        signals.append({"icon":"🔗","type":"warn","text":"Cross-verify with multiple trusted sources"})

    # Fallback
    if not signals:
        if verdict == "FAKE":
            signals = [{"icon":"⚠️","type":"danger","text":"Matches known fake news language patterns"},
                       {"icon":"📚","type":"danger","text":"No factual references or verified sources"}]
        else:
            signals = [{"icon":"✅","type":"ok","text":"Neutral, factual language detected"},
                       {"icon":"🔗","type":"warn","text":"Always verify with trusted news sources"}]
    return signals[:5]

# ── Linguistic metrics ────────────────────────
def get_metrics(text, verdict, conf):
    t     = text.lower()
    total = max(len(t.split()), 1)
    sens  = min(100, int(sum(t.count(w) for w in FAKE_WORDS) / total * 800))
    emo   = min(100, int(sum(t.count(w) for w in EMO_WORDS)  / total * 700))
    fact  = min(100, int(sum(t.count(w) for w in REAL_WORDS) / total * 900))
    click = min(100, text.count("!")*8 + sum(1 for w in text.split() if w.isupper() and len(w)>2)*12)
    # Adjust to match verdict
    if verdict == "FAKE" and conf > 70:
        sens = max(sens,45); emo = max(emo,40); fact = min(fact,35); click = max(click,40)
    elif verdict == "REAL" and conf > 70:
        sens = min(sens,30); emo = min(emo,25); fact = max(fact,55); click = min(click,25)
    return [{"label":"Sensationalism","value":sens}, {"label":"Emotional language","value":emo},
            {"label":"Factual references","value":fact}, {"label":"Clickbait score","value":click}]

# ── Detailed reason ───────────────────────────
def get_reason(text, verdict, conf, signals):
    danger = [s["text"] for s in signals if s["type"] == "danger"]
    ok     = [s["text"] for s in signals if s["type"] == "ok"]
    parts  = []

    if verdict == "FAKE":
        if any("sensational" in s.lower() for s in danger):
            parts.append("sensational and exaggerated language is used")
        if any("credible" in s.lower() or "source" in s.lower() for s in danger):
            parts.append("no credible source or official attribution was found")
        if any("emotional" in s.lower() for s in danger + [s["text"] for s in signals if s["type"]=="warn"]):
            parts.append("emotional trigger words are used to provoke the reader")
        if not parts:
            parts = ["the language pattern matches known fake news writing style"]
        conf_note = f" Model confidence: {conf}%. Do not share without verifying."
        return "This article is predicted as FAKE because " + "; ".join(parts) + "." + conf_note
    else:
        if any("credible" in s.lower() or "reference" in s.lower() for s in ok):
            parts.append("credible source references are present")
        if not parts:
            parts = ["the language is neutral and factual, consistent with real journalism"]
        conf_note = f" Model confidence: {conf}%. Always cross-verify with trusted sources."
        return "This article is predicted as REAL because " + "; ".join(parts) + "." + conf_note

# ── Source credibility ────────────────────────
SOURCES = {"bbc.com":96,"reuters.com":97,"apnews.com":97,"thehindu.com":91,
           "ndtv.com":84,"timesofindia.com":82,"who.int":98,"nasa.gov":97,
           "pib.gov.in":97,"indianexpress.com":87,"infowars.com":5,"naturalnews.com":8}

def check_source(url):
    if not url: return []
    try:
        from urllib.parse import urlparse
        domain = urlparse(url).netloc.lower().replace("www.","")
        score  = SOURCES.get(domain, 45)
        note   = "Known trusted source" if score > 75 else ("Low credibility" if score < 20 else "Unknown domain")
        return [{"name":domain,"score":score,"note":note}]
    except: return []

# ── Core prediction ───────────────────────────
def predict_news(text, lang="en", url="", source_type="text"):
    cleaned = clean(text)
    vec     = tfidf.transform([cleaned])
    label   = model.predict(vec)[0]
    proba   = model.predict_proba(vec)[0]
    verdict = "FAKE" if label == 0 else "REAL"
    conf    = round(float(proba[label]) * 100, 1)
    signals = get_signals(text, verdict)

    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO predictions (text,lang,verdict,conf,source,created) VALUES (?,?,?,?,?,?)",
                 (text[:150], lang, verdict, conf, source_type, datetime.datetime.now().isoformat()))
    conn.commit(); conn.close()

    return {"verdict":verdict, "confidence":conf, "keywords":get_keywords(cleaned),
            "signals":signals, "metrics":get_metrics(text, verdict, conf),
            "source_credibility":check_source(url),
            "reason":get_reason(text, verdict, conf, signals), "message":""}

# ── Routes ────────────────────────────────────
@app.route("/")
def home(): return send_from_directory(FOLDER, "index.html")

@app.route("/<page>.html")
def pages(page): return send_from_directory(FOLDER, f"{page}.html")

@app.route("/predict", methods=["POST"])
def predict():
    body = request.json or {}
    news = body.get("text","").strip()
    lang = body.get("lang","en")
    url  = body.get("url","").strip()

    if len(news.split()) < 20:
        return jsonify({"verdict":"TOO_SHORT","confidence":0,"keywords":[],
                        "message":f"Only {len(news.split())} words. Need at least 20."})

    translated = translate(news, lang)
    result     = predict_news(translated or news, lang, url, "text")
    result["translated_text"] = translated
    return jsonify(result)

@app.route("/analyze_video", methods=["POST"])
def analyze_video():
    if "file" not in request.files:
        return jsonify({"error":"No file uploaded"}), 400
    f   = request.files["file"]
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in [".mp4",".avi",".mov",".mkv",".mp3",".wav",".m4a"]:
        return jsonify({"error":"Unsupported file type"}), 400

    tmp_v = os.path.join(FOLDER, "tmp_video" + ext)
    tmp_a = os.path.join(FOLDER, "tmp_audio.wav")
    f.save(tmp_v)
    try:
        if ext == ".wav":
            import shutil; shutil.copy(tmp_v, tmp_a)
        elif ext in [".mp3",".m4a"]:
            from pydub import AudioSegment
            AudioSegment.from_file(tmp_v).export(tmp_a, format="wav")
        else:
            from moviepy.editor import VideoFileClip
            clip = VideoFileClip(tmp_v)
            clip.audio.write_audiofile(tmp_a, verbose=False, logger=None)
            clip.close()

        import speech_recognition as sr
        r = sr.Recognizer()
        with sr.AudioFile(tmp_a) as src:
            transcript = r.recognize_google(r.record(src))

        if len(transcript.split()) < 10:
            return jsonify({"error":"Not enough speech detected in video"}), 400

        result = predict_news(transcript, "en", "", "video")
        result["transcript"] = transcript
        return jsonify(result)
    except ImportError:
        return jsonify({"error":"Run: pip install moviepy SpeechRecognition pydub"}), 500
    except Exception as e:
        return jsonify({"error":str(e)}), 500
    finally:
        for fp in [tmp_v, tmp_a]:
            try: os.remove(fp)
            except: pass

@app.route("/analyze_youtube", methods=["POST"])
def analyze_youtube():
    url = (request.json or {}).get("url","").strip()
    if not url or ("youtube.com" not in url and "youtu.be" not in url):
        return jsonify({"error":"Please provide a valid YouTube URL"}), 400

    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        m = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
        if not m: return jsonify({"error":"Could not extract video ID"}), 400
        vid_id = m.group(1)
        try:    t_list = YouTubeTranscriptApi.get_transcript(vid_id, languages=["en"])
        except: t_list = YouTubeTranscriptApi.get_transcript(vid_id)
        transcript = " ".join([t["text"] for t in t_list])
    except ImportError:
        return jsonify({"error":"Run: pip install youtube-transcript-api"}), 500
    except Exception:
        return jsonify({"error":"No captions found. Try a video with subtitles enabled."}), 400

    if len(transcript.split()) < 10:
        return jsonify({"error":"Transcript too short to analyze"}), 400

    result = predict_news(transcript[:3000], "en", url, "youtube")
    result["transcript"]       = transcript[:1000]
    result["transcript_words"] = len(transcript.split())
    return jsonify(result)

@app.route("/stats")
def stats():
    conn = sqlite3.connect(DB_PATH)
    rows = dict(conn.execute("SELECT verdict, COUNT(*) FROM predictions GROUP BY verdict").fetchall())
    src  = dict(conn.execute("SELECT source, COUNT(*) FROM predictions GROUP BY source").fetchall())
    conn.close()
    return jsonify({"total":rows.get("FAKE",0)+rows.get("REAL",0),
                    "fake":rows.get("FAKE",0), "real":rows.get("REAL",0), "by_source":src})

@app.route("/history")
def get_history():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT text,lang,verdict,conf,source,created FROM predictions ORDER BY id DESC LIMIT 50").fetchall()
    conn.close()
    return jsonify([{"text":r[0],"lang":r[1],"verdict":r[2],"conf":r[3],"source":r[4],"created":r[5]} for r in rows])

@app.route("/clear", methods=["POST"])
def clear():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM predictions"); conn.commit(); conn.close()
    return jsonify({"ok":True})

if __name__ == "__main__":
    print("FakeNewsIQ v3.0 — http://127.0.0.1:5000")
    app.run(debug=True)