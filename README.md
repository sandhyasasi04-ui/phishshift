# PhishShift Message Risk Scanner (ML-07)
Flask dashboard + scikit-learn models for triaging simulated phishing messages. Synthetic data only; links are defanged and never opened.

## Run locally (VS Code terminal)
```
python -m venv .venv
.venv\Scripts\activate        # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python train.py               # builds corpus, trains 3 models, writes data/metrics.json
python app.py                 # open http://127.0.0.1:5000
python -m unittest tests.test_core -v
```

## GitHub, then Render
```
git init && git add . && git commit -m "PhishShift ML-07"
git branch -M main && git remote add origin <your-repo-url> && git push -u origin main
```
Render: New > Blueprint (reads render.yaml) or New > Web Service with
Build `pip install -r requirements.txt && python train.py`, Start `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`.

## Files
- core.py: corpus generator, dedup, rules, models, perturbations
- train.py: split by campaign, metrics, saved bundle
- app.py + templates/index.html: dashboard and API
- data/: raw_corpus.csv, clean_corpus.csv, metrics.json (provenance: locally authored synthetic simulations)
- tests/test_core.py: checks for FR-01, FR-03, FR-04, NFR-01

## Still yours to write (SRS section 13)
SRS.pdf, final_report.pdf, screenshots, 3-5 minute demo, experiment journal, AI-usage disclosure, threat model, and a mentor-supplied campaign for the final live variation. Be ready to explain every file.
