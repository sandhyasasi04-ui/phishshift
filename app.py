import json, os, joblib
from flask import Flask, jsonify, request, render_template
import core

app = Flask(__name__)
if not os.path.exists("models/bundle.joblib"):
    import train; train.main()
B = joblib.load("models/bundle.joblib"); MODELS, PRIMARY = B["models"], B["primary"]
WORD = next(m for m in MODELS if m.name.startswith("TF-IDF"))

def scan(text):
    ps = {m.name: float(m.proba([text])[0]) for m in MODELS}; spread = max(ps.values()) - min(ps.values())
    p = ps[PRIMARY]
    return dict(text=text, probs=ps, primary=PRIMARY, p=p, spread=spread, verdict=core.verdict(p, spread),
                cues=core.cues(text), terms=WORD.terms(text), url=core.url_features(text))

@app.get("/")
def index(): return render_template("index.html")

@app.get("/healthz")
def health(): return "ok"

@app.post("/api/scan")  # text is only analysed as a string: links are never opened or fetched (NFR-01)
def api_scan():
    d = request.get_json(silent=True) or {}
    text = core.join(str(d.get("subject", ""))[:300], str(d.get("body", ""))[:4000])
    if len(text) < 8: return jsonify(error="Paste a message of at least a few words."), 400
    out = scan(text); k = d.get("perturb")
    if k in core.PERTS: out["perturbed"] = scan(core.perturb(text, k))
    return jsonify(out)

@app.get("/api/metrics")
def api_metrics(): return jsonify(json.load(open("data/metrics.json")))

@app.get("/api/samples")
def api_samples(): return jsonify([dict(text=t, truth=y) for t, y in core.CP5])

@app.get("/api/cp5")
def api_cp5():
    rows = []
    for t, y in core.CP5:
        r = dict(text=t, truth=y)
        for k in ["base"] + core.PERTS:
            s = scan(core.perturb(t, k) if k != "base" else t); r[k] = dict(p=s["p"], verdict=s["verdict"])
        rows.append(r)
    return jsonify(rows)

if __name__ == "__main__": app.run(debug=True, port=5000)
