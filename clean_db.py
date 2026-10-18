import re, shutil
import chromadb

shutil.copytree("data/vector_db", "data/vector_db_backup", dirs_exist_ok=True)

col = chromadb.PersistentClient(path="data/vector_db").get_collection("agriculture")

STOP = set("the of and in to is for are with a an on as by be or from that this it at can should "
           "plant crop soil water seed seeds field farmers yield land".split())

def script_ratio(t, lo, hi):
    letters = [c for c in t if c.isalpha()]
    return sum(lo <= c <= hi for c in letters) / max(1, len(letters))

def is_good(t):
    t = t.strip()
    if len(t) < 40:
        return False
    if script_ratio(t, "\u0C80", "\u0CFF") > 0.5:      # Kannada
        return True
    if script_ratio(t, "\u0900", "\u097F") > 0.5:      # Hindi
        return True
    words = re.findall(r"[A-Za-z]+", t.lower())
    if not words:
        return False
    ratio = sum(w in STOP for w in words) / len(words)
    junk = len(re.findall(r"[A-Za-z][;\[\]\^][A-Za-z]", t)) / max(1, len(words))
    return ratio >= 0.15 and junk < 0.02

data = col.get(include=["documents"])
bad = [i for i, d in zip(data["ids"], data["documents"]) if not is_good(d)]
print(f"Total: {len(data['ids'])}  Junk: {len(bad)}")

for i, d in list(zip(data["ids"], data["documents"]))[:0]:
    pass
print("Sample junk:")
shown = 0
for i, d in zip(data["ids"], data["documents"]):
    if i in set(bad) and shown < 5:
        print("-", d[:100].replace("\n", " "))
        shown += 1

if input("Delete junk chunks? (y/n): ").lower() == "y":
    for k in range(0, len(bad), 500):
        col.delete(ids=bad[k:k + 500])
    print("Done. Remaining:", col.count())