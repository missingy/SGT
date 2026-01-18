import argparse
import os
import csv

DATA_DIR = r"data/Data_100"
OUT_CSV = r"artifacts/midi-embs/myset/labels.csv"


def parse_emotion_genre(fname: str):
    name = os.path.splitext(os.path.basename(fname))[0]
    parts = name.split("_")
    if len(parts) < 4:
        return None, None
    return parts[1].lower(), parts[2].lower()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=DATA_DIR, help="Dataset root directory")
    ap.add_argument("--out-csv", default=OUT_CSV, help="Output labels.csv")
    args = ap.parse_args()

    rows = []
    for root, _, files in os.walk(args.data_root):
        for fname in files:
            if not fname.lower().endswith((".mid", ".midi")):
                continue
            emo, gen = parse_emotion_genre(fname)
            if emo and gen:
                rows.append([fname, emo, gen])

    rows.sort()

    if os.path.dirname(args.out_csv):
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["piece", "emotion", "genre"])
        w.writerows(rows)

    print("[OK] labels:", args.out_csv, "N=", len(rows))


if __name__ == "__main__":
    main()
