"""PhishShift core: synthetic corpus, dedup, rules, ML models, perturbations. Nothing here opens or fetches a link."""
import re, math, random, datetime as dt
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

HOLDOUT = {"gift_card_ceo", "vendor_bank_change", "payroll_notice", "it_maintenance", "parcel_ok"}
URGENT_LEGIT = {"pw_reset_notice", "payroll_notice", "it_maintenance"}
LO, HI = 0.35, 0.65  # abstain band

# campaign: label, channel, subjects, openers, cores, asks, closers  ("|" separated pools)
CAMP = {
 "payroll_update": ("phish","email","Action required: payroll update|Your salary deposit is on hold|Payroll profile verification","Dear employee,|Hello {n},|Attention {n}:","Your next salary payment is on hold because your payroll profile is incomplete.|HR flagged a mismatch in your direct deposit details.|We could not credit your salary due to an account verification issue.","Verify your password and bank details here: {u}|Confirm your login and update your deposit details: {u}|Sign in to keep your payment active: {u}","Failure to respond within {h} hours will delay your pay.|This link expires today.|Reply immediately to avoid penalties."),
 "mfa_reset": ("phish","email","Your account will be locked|Security alert: new sign-in|MFA device expiring","Dear user,|Hello {n},|Security notice:","We detected a sign-in from a new device.|Your multi-factor authentication token expires soon.|Unusual activity was found on your mailbox.","Re-enter your password to keep access: {u}|Confirm your credentials now: {u}|Verify your identity: {u}","Your account will be suspended in {h} hours.|Act now to avoid lockout.|Final notice."),
 "parcel_fee": ("phish","sms","","Dear customer,|Hi {n},|Notice:","Your parcel is held at the depot.|Delivery failed: address incomplete.|A customs fee is pending for your package.","Pay the small release fee: {u}|Reschedule delivery and pay {amt}: {u}|Confirm your card details: {u}","Expires in {h} hours.|Final notice before return.|Act now."),
 "invoice_overdue": ("phish","email","Overdue invoice #{i}|Final reminder: invoice unpaid|Payment failed for invoice {i}","Dear customer,|Hello {n},|Accounts team:","Our records show invoice {i} for {amt} is overdue.|Your last payment was declined.|We have not received the wire transfer for {i}.","View and pay the invoice: {u}|Download the statement here: {u}|Confirm your bank details: {u}","Legal action may follow after {h} hours.|Please settle immediately.|Late penalties apply."),
 "gift_card_ceo": ("phish","email","Quick favour|Are you at your desk?|Need your help today","Hi {n},|{n},|Hey {n},","I'm stuck in a board meeting and can't take calls.|I want to surprise a few clients this week.|I need something sorted before lunch.","Could you pick up {k} vouchers from the store and send me the codes?|Grab some store credit cards and photograph the backs for me.|Can you buy a few prepaid codes and message the numbers back?","Keep this between us for now.|I'll reimburse you later today.|Send them over quietly, thanks."),
 "vendor_bank_change": ("phish","email","Updated remittance details|Banking change for next invoice|Please note new account","Hello {n},|Dear partner,|Hi {n},","Our finance team has moved to a new bank this quarter.|Following an audit we changed our remittance account.|Our old account is being closed next month.","Please route the next payment to the account in this form: {u}|Kindly update your records using this portal: {u}|Confirm the change by replying with your approval.","Do not discuss this with our old contact.|Thank you for the quick turnaround.|Please treat this as time-sensitive."),
 "pw_reset_notice": ("legit","email","Your password was changed|Password reset requested|Password expires in {h} days","Hi {n},|Hello {n},|Good morning {n},","You asked to reset your password a moment ago.|Your password was updated successfully.|Your password will expire soon according to company policy.","If this was you, no action is needed.|To choose a new password, open the IT portal from your bookmarks.|Visit the internal portal at {ui} if you need help.","If you did not request this, contact the helpdesk on ext. 4400.|This is an automated notice.|Thanks, IT Service Desk"),
 "payroll_notice": ("legit","email","Payslip for {m} available|Payroll cut-off reminder|Urgent: timesheets due today","Hi {n},|Hello {n},|Dear colleague,","Your payslip is ready in the HR system.|Timesheets must be submitted before the payroll cut-off.|Payroll will run on {d}, please check your hours.","Log in through the usual HR portal to view it.|Submit your timesheet by 5 pm today.|No bank details are needed; do not share them.","Contact payroll@company.example with questions.|Thanks, People Operations|This is an urgent reminder for approvers."),
 "it_maintenance": ("legit","email","Urgent: server maintenance tonight|Planned VPN outage|Immediate action: save your work","Hi all,|Hello {n},|Team,","The file server will be offline from 22:00 for patching.|VPN access will be unavailable for {h} hours.|Please close open files before the maintenance window.","No action beyond saving your work is required.|Check the status page at {ui} for updates.|Reach the helpdesk if your work is affected.","Thanks for your patience.|This notice is being sent to all staff.|Service will resume by {d}."),
 "hr_benefits": ("legit","email","Open enrolment starts {d}|Benefits summary|Wellness programme update","Hi {n},|Hello {n},|Good morning,","Open enrolment for benefits begins next week.|The updated benefits guide is on the intranet.|New wellness options are now available.","Read the guide at {ui} and pick your plan by the deadline.|Book a short session with HR if you have questions.|No forms are required this year.","Thanks, HR team|Deadline is end of the month.|Please read before Friday."),
 "meeting_invite": ("legit","email","Team sync moved to {d}|Agenda for project review|Lunch and learn","Hi {n},|Hello team,|Hey {n},","The weekly sync has been rescheduled.|Please review the agenda before the meeting.|We will cover the roadmap and open risks.","Join using the calendar invite.|Add your topics to the shared doc by {d}.|Let me know if the new time does not work.","See you there.|Thanks,|Cheers"),
 "parcel_ok": ("legit","sms","","Hi {n},|Hello,|Update:","Your order {i} has shipped and arrives {d}.|Delivery update: your package is out for delivery.|Your parcel was delivered to the front desk.","Track it in the retailer app.|No payment is needed.|Sign for it at reception.","Reply STOP to opt out.|Thanks for shopping with us.|Questions? Call the number on your receipt."),
}
SENDERS = {"phish": ["lookalike_domain","free_mail","unknown_external","spoofed_display_name"], "legit": ["internal","known_vendor","verified_service"]}
NAMES = "Sam Priya Alex Dana Ravi Mei Omar Lena Tom Aisha".split()
DAYS = "Monday Tuesday Wednesday Thursday Friday".split()
MONTHS = "September October November".split()

def make(n_per=90, seed=7):
    R, rows, k = random.Random(seed), [], 0
    for cid, (lab, chan, sj, op, co, ak, cl) in CAMP.items():
        P = [x.split("|") for x in (sj, op, co, ak, cl)]
        for _ in range(n_per):
            r = "".join(R.choices("abcdefghjkmnpqrstuvwxyz23456789", k=5))
            f = dict(n=R.choice(NAMES), h=R.choice([2,4,6,12,24,48]), d=R.choice(DAYS), i=R.randint(1000,9999), m=R.choice(MONTHS),
                     amt=R.choice(["$1.99","$4.20","$250","$1,240"]), k=R.choice(["three","five","two"]),
                     ui="hxxps://intranet[.]company[.]example/" + R.choice(["it","hr","help"]),
                     u=R.choice([f"hxxps://bit[.]ly/3{r}", f"hxxp://secure-{r}[.]xyz/login", f"hxxps://{r}-support[.]top/verify", f"hxxps://rb[.]gy/{r}"]))
            subj, o, c, a, z = [R.choice(p).format(**f) for p in P]
            body = " ".join([o, c, a, z]).strip()
            rows.append(dict(message_id=f"M{k:05d}", channel=chan, subject_or_title=subj, body_text=body,
                sanitized_url_features=str(url_features(body)), sender_domain_type=R.choice(SENDERS[lab]),
                label=lab, campaign_group=cid, collected_date=str(dt.date(2026,9,1) + dt.timedelta(R.randint(0,44))),
                source="locally authored simulation (synthetic, CC0)"))
            k += 1
    df = pd.DataFrame(rows)
    df["text"] = [join(s, b) for s, b in zip(df.subject_or_title, df.body_text)]
    df["y"] = (df.label == "phish").astype(int)
    return df

def join(subject, body): return f"{subject.strip()}. {body.strip()}" if subject.strip() else body.strip()

def dedup(df, thr=0.88):
    v = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(df["text"]); S = (v @ v.T).toarray()
    keep, dropped = [], []
    for i in range(len(df)):
        j = next((k for k in keep if df.label.iat[k] == df.label.iat[i] and S[i, k] >= thr), None)
        if j is None: keep.append(i)
        else: dropped.append(i)
    return df.iloc[keep].reset_index(drop=True), df.iloc[dropped]

SHORT = r"(bit\[?\.\]?ly|tinyurl\[?\.\]?com|rb\[?\.\]?gy|cutt\[?\.\]?ly|is\[?\.\]?gd)"
ODD = r"\[?\.\]?(xyz|top|click|icu|work)\b"
def url_features(t):
    links = re.findall(r"(?:hxxps?|https?)(?::|\[:\])//\S+", t); s = " ".join(links).lower()
    return dict(n_links=len(links), shortener=int(bool(re.search(SHORT, s))), odd_tld=int(bool(re.search(ODD, s))),
                defanged=int("[.]" in s or "hxxp" in s), internal=int("company[.]example" in s))
def prep(t):
    f = url_features(t)
    return t + " " + " ".join([f"__links{min(f['n_links'],2)}"] + [f"__{k}" for k in ("shortener","odd_tld","defanged","internal") if f[k]])

CUES = [
 ("Pressure to act fast", r"\b(urgent|immediately|within \d+ hours?|expires? (today|soon)|final notice|act now|before 5 pm|time-sensitive)\b", 1.0),
 ("Asks for login or bank details", r"\b(verify your (password|identity)|confirm your (login|credentials|card details|bank details)|re-?enter your password|bank details|sign in to keep|type your login)\b", 1.6),
 ("Shortened or odd link", SHORT + r"\S*|\S*" + ODD + r"\S*", 1.4),
 ("Money, vouchers or fees", r"\b(gift cards?|wire transfer|prepaid|release fee|handling fee|customs fee|credit cards?|vouchers?|remittance)\b", 1.0),
 ("Threat of loss", r"\b(suspended|locked|lockout|penalt(y|ies)|legal action|stop deletion)\b", 1.1),
 ("Generic greeting", r"\b(dear (customer|user|employee)|team member)\b", 0.7),
 ("Secrecy request", r"\b(keep this between us|do not discuss|quietly)\b", 1.2),
]
def cues(t):
    out = []
    for lab, pat, w in CUES:
        for m in re.finditer(pat, t, re.I): out.append(dict(label=lab, start=m.start(), end=m.end(), weight=w))
    return sorted(out, key=lambda c: c["start"])
def cue_score(cs):
    z = sum({c["label"]: c["weight"] for c in cs}.values()) - 1.8
    return 1 / (1 + math.exp(-1.6 * z))

class Rules:
    name = "Keyword rules"
    def fit(self, t, y): return self
    def proba(self, texts): return np.array([cue_score(cues(t)) for t in texts])

class ML:
    def __init__(self, name, vec):
        self.name = name; self.pipe = make_pipeline(vec, LogisticRegression(C=4, max_iter=3000, class_weight="balanced"))
    def fit(self, t, y): self.pipe.fit([prep(x) for x in t], y); return self
    def proba(self, texts): return self.pipe.predict_proba([prep(x) for x in texts])[:, 1]
    def terms(self, t, k=5):
        vec, clf = self.pipe.steps[0][1], self.pipe.steps[1][1]
        x = vec.transform([prep(t)]); c = np.asarray(x.multiply(clf.coef_[0]).todense()).ravel(); names = vec.get_feature_names_out()
        o = np.argsort(c); return dict(risk=[names[i] for i in o[::-1][:k] if c[i] > 0], safe=[names[i] for i in o[:k] if c[i] < 0])

def build_models():
    return [Rules(), ML("TF-IDF word LR", TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True, min_df=2)),
            ML("Char n-gram LR", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), sublinear_tf=True, min_df=2))]

def perturb(t, kind):
    if kind == "punct": return t.replace(",", "").replace(". ", " ... ").replace(":", " -") + " !!"
    if kind == "defang": return t.replace("[.]", " dot ").replace("hxxp", "hXXp")
    if kind == "case": return " ".join(w.capitalize() for w in t.split())
    return t
PERTS = ["punct", "defang", "case"]

def verdict(p, spread):
    if LO < p < HI or spread > 0.45: return "review"
    return "flag" if p >= HI else "clear"

CP5 = [("Hello team member, your mailbox is over quota. Sign in at hxxp://mail-storage[.]top/renew to stop deletion in 6 hours.", 1),
 ("Payroll alert: your last payslip failed. Confirm your bank details at hxxps://rb[.]gy/pay77 before 5 pm.", 1),
 ("Hey, are you free? I need a few store credit cards for a client gift, and please keep it between us.", 1),
 ("Courier notice: your package is held. Pay a $1.50 handling fee at hxxps://tinyurl[.]com/hold12 today.", 1),
 ("Hi Priya, your password was reset at 09:14. If this was you, nothing more to do. If not, call the helpdesk.", 0),
 ("Payroll runs tomorrow. Please approve your team's timesheets in the HR system by 4 pm today.", 0),
 ("Urgent: the printer on floor 3 is out of service until noon. Please use floor 2 meanwhile.", 0),
 ("Your order 5521 was delivered to reception. No payment needed. Thanks for shopping with us.", 0)]
