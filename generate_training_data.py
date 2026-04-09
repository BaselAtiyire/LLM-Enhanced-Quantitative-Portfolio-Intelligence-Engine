import json, os, random, pandas as pd

def load_ranking_csv(f):
    return pd.read_csv(f, encoding="utf-8-sig").to_dict(orient="records")

def sf(v):
    try: return float(v)
    except: return 0.0

def fmt(rows):
    L = ["RANKING DATASET (use ONLY this data):"]
    for r in rows:
        try:
            L.append(str(r.get("rank","")) + " " + str(r.get("ticker","")) +
                " score=" + str(round(sf(r.get("score")),3)) +
                " sharpe=" + str(round(sf(r.get("sharpe")),3)) +
                " 1w=" + str(round(sf(r.get("one_week_pct")),2)) +
                "% vol=" + str(round(sf(r.get("vol_ann"))*100,2)) +
                "% pe=" + str(round(sf(r.get("trailing_pe")),1)))
        except: pass
    return "\n".join(L)

def gen(rows):
    sr = sorted(rows, key=lambda x: int(x.get("rank",99)))
    if not sr: return []
    t = sr[0]
    bot = sr[-1]
    ctx = fmt(rows)
    return [
        (ctx, "Why is " + t["ticker"] + " ranked first?",
         t["ticker"] + " score=" + str(round(sf(t.get("score")),3)) +
         " sharpe=" + str(round(sf(t.get("sharpe")),3)) +
         " 1w=" + str(round(sf(t.get("one_week_pct")),2)) + "%. From dataset only."),
        (ctx, "Why is " + bot["ticker"] + " ranked last?",
         bot["ticker"] + " score=" + str(round(sf(bot.get("score")),3)) +
         " vol=" + str(round(sf(bot.get("vol_ann"))*100,2)) + "%. From dataset only."),
        (ctx, "What will the price be next week?",
         "Cannot predict future prices. Only dataset values are allowed."),
        (ctx, "What is the EPS?",
         "EPS not in dataset. Available metrics: score, returns, volatility, Sharpe, drawdown, PE, market cap."),
        (ctx, "How is the score calculated?",
         "Score equals sum of w_k times z_ik. Weights: market_cap 0.30, 1w 0.25, 30d 0.15, pe 0.15 inverted, vol 0.15 inverted."),
    ]

def main():
    csvs = [f for f in os.listdir(".") if f.startswith("financial_analysis") and f.endswith(".csv")]
    if not csvs:
        print("No CSV files found")
        return
    all_p = []
    for f in csvs:
        print("Processing " + f)
        try:
            rows = load_ranking_csv(f)
            for ctx, q, a in gen(rows):
                all_p.append({
                    "instruction": "Answer using ONLY the dataset. Do not hallucinate.",
                    "input": ctx + "\n\nQuestion: " + q,
                    "output": a
                })
            print("  " + str(len(gen(rows))) + " pairs")
        except Exception as e:
            print("  Error: " + str(e))
    if not all_p:
        print("No pairs generated")
        return
    random.shuffle(all_p)
    sp = int(len(all_p) * 0.9)
    train = all_p[:sp]
    evl = all_p[sp:] if all_p[sp:] else all_p[:1]
    with open("hpie_train.jsonl", "w", encoding="utf-8") as f:
        for x in train:
            f.write(json.dumps(x) + "\n")
    with open("hpie_eval.jsonl", "w", encoding="utf-8") as f:
        for x in evl:
            f.write(json.dumps(x) + "\n")
    print("Done. Train=" + str(len(train)) + " Eval=" + str(len(evl)))
    print("Files created: hpie_train.jsonl  hpie_eval.jsonl")

main()
