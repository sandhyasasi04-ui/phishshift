"""Builds corpus, dedups, trains/evaluates 3 models with campaign holdout, writes metrics + model bundle."""
import json, os, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import core

def ev(m, df):
    p = m.proba(df.text.tolist()); pred = p >= .5; y = df.y.values; tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum()); u = df.campaign_group.isin(core.URGENT_LEGIT).values
    pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(acc=(tp + tn) / len(df), precision=pr, recall=rc, f1=2 * pr * rc / max(pr + rc, 1e-9), cm=[[tn, fp], [fn, tp]],
                fpr_urgent=float(pred[u].mean()) if u.any() else None, abstain=float(((p > core.LO) & (p < core.HI)).mean()))

def drift(m, df):
    t = df.text.tolist(); p0 = m.proba(t); out = {}
    for k in core.PERTS:
        p1 = m.proba([core.perturb(x, k) for x in t]); out[k] = dict(flip=float(((p0 >= .5) != (p1 >= .5)).mean()), shift=float(np.abs(p1 - p0).mean()))
    return out

def main():
    os.makedirs("data", exist_ok=True); os.makedirs("models", exist_ok=True)
    raw = core.make(); raw.drop(columns=["text", "y"]).to_csv("data/raw_corpus.csv", index=False)
    df, dropped = core.dedup(raw); df.drop(columns=["text", "y"]).to_csv("data/clean_corpus.csv", index=False)
    bad = int(df.body_text.str.contains(r"https?://", regex=True).sum())  # NFR-01: every URL must be defanged
    te = df[df.campaign_group.isin(core.HOLDOUT)]; tr = df[~df.campaign_group.isin(core.HOLDOUT)]
    v = TfidfVectorizer(ngram_range=(1, 2)).fit(df.text); sim = float((v.transform(te.text) @ v.transform(tr.text).T).max())
    rtr, rte = train_test_split(df, test_size=.25, stratify=df.y, random_state=7)
    M = {"dataset": dict(raw=len(raw), clean=len(df), removed=len(dropped), phish=int(df.y.sum()), legit=int((1 - df.y).sum()), train=len(tr), holdout=len(te),
            campaigns=df.groupby("campaign_group").size().to_dict(), holdout_campaigns=sorted(core.HOLDOUT), live_links=bad, max_train_test_similarity=sim, dedup_threshold=.88), "models": {}}
    final = []
    for i, m in enumerate(core.build_models()):
        r = core.build_models()[i].fit(rtr.text.tolist(), rtr.y.values); rnd = ev(r, rte)
        m.fit(tr.text.tolist(), tr.y.values); h = ev(m, te); final.append(m)
        rec = {c: float((m.proba(g.text.tolist()) >= .5).mean()) for c, g in te[te.y == 1].groupby("campaign_group")}
        M["models"][m.name] = dict(random=rnd, holdout=h, drift=drift(m, te), phish_recall_by_campaign=rec)
    primary = max(M["models"], key=lambda k: (M["models"][k]["holdout"]["f1"], -(M["models"][k]["holdout"]["fpr_urgent"] or 0))); M["primary"] = primary
    at1 = "Hi Sam, your password was changed today. If this was you, no action is needed. If not, please call the helpdesk on ext. 4400."
    at3 = core.CP5[1][0]
    M["acceptance"] = dict(
        AT01=dict(text=at1, probs={m.name: float(m.proba([at1])[0]) for m in final}, fpr_urgent={k: v["holdout"]["fpr_urgent"] for k, v in M["models"].items()}),
        AT02={k: v["phish_recall_by_campaign"] for k, v in M["models"].items()},
        AT03=dict(text=at3, shifts={m.name: {"base": float(m.proba([at3])[0]), **{k: float(m.proba([core.perturb(at3, k)])[0]) for k in core.PERTS}} for m in final}))
    json.dump(M, open("data/metrics.json", "w"), indent=1); joblib.dump(dict(models=final, primary=primary), "models/bundle.joblib")
    print(f"corpus {len(raw)} -> {len(df)} after dedup | primary: {primary}")
    for k, v in M["models"].items(): print(f"{k:16s} random F1 {v['random']['f1']:.2f} | holdout F1 {v['holdout']['f1']:.2f} | FPR urgent-legit {v['holdout']['fpr_urgent']:.2f}")

if __name__ == "__main__": main()
