import time
from pathlib import Path
import requests

INBOX = Path("D:/LocalAI/inbox")
INBOX.mkdir(parents=True, exist_ok=True)
PROCESSED = INBOX / "processed"
PROCESSED.mkdir(exist_ok=True)

BRAIN_URL = "http://localhost:8000/run_project"

print(f"👀 Watching {INBOX} for new .txt files...")
seen = set(INBOX.glob("*.txt"))

while True:
    current = set(INBOX.glob("*.txt"))
    new_files = current - seen
    for f in new_files:
        print(f"📥 New file: {f.name}")
        try:
            with open(f, "r", encoding="utf-8") as fh:
                goal = fh.read().strip()
            if not goal:
                print("⚠️  Empty file, skipping.")
                f.rename(PROCESSED / f.name)
                continue
            resp = requests.post(
                BRAIN_URL,
                json={"goal": goal, "max_tasks": 5},
                timeout=900
            )
            if resp.status_code == 200:
                print(f"✅ Brain triggered. Summary: {resp.json().get('final_summary', '')[:200]}...")
                # Move to processed so we don't re-run
                f.rename(PROCESSED / f.name)
            else:
                print(f"❌ Brain returned {resp.status_code}: {resp.text}")
                # Keep file for retry (or move to an error folder)
        except Exception as e:
            print(f"🔥 Error processing {f.name}: {e}")
    seen = current
    time.sleep(10)   # check every 10 seconds
