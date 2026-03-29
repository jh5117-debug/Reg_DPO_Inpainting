#!/usr/bin/env bash
set -euo pipefail

export PROJECT_ROOT=/sc-projects/sc-proj-cc09-repair/hongyou/dev/Reg_DPO_Inpainting
export DATA_ROOT=$PROJECT_ROOT/data/DPO_Finetune_data
export STAGE_ROOT=$PROJECT_ROOT/.hf_repair_stage_dirs
export HF_HOME=$PROJECT_ROOT/.hf_cache_repair
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTHONUNBUFFERED=1
export REPO_ID=JiaHuang01/DPO-dataset
export REVISION=a0cfd326c120d71fa3ade5f56f830fa2318eb903

cd "$PROJECT_ROOT"
mkdir -p "$STAGE_ROOT" "$HF_HOME"

echo "=== Step 0: HF auth check ==="
if ! hf auth whoami >/dev/null 2>&1; then
  echo "HF login not found. Please login now."
  hf auth login
fi
hf auth whoami

echo "=== Step 1: preflight remote check for the 17 affected video dirs ==="
python - <<'PY'
import os
from huggingface_hub import HfApi

repo_id = os.environ["REPO_ID"]
videos = [
    "ytbv_38fe9b3ed1",
    "ytbv_6f96e91d81",
    "ytbv_73e4e5cc74",
    "ytbv_cb2e35d610",
    "ytbv_0503bf89c9",
    "ytbv_aeba9ac967",
    "ytbv_45d9fc1357",
    "ytbv_045f00aed2",
    "ytbv_72cae058a5",
    "ytbv_3b6e983b5b",
    "ytbv_04f21da964",
    "ytbv_6d12e30c48",
    "ytbv_804f6338a4",
    "ytbv_a5ec5b0265",
    "ytbv_df365282c6",
    "davis_dogs-scale",
    "ytbv_da6c68708f",
]

api = HfApi()
missing = []
for video in videos:
    items = list(api.list_repo_tree(repo_id=repo_id, repo_type="dataset", path_in_repo=video, recursive=True))
    file_count = sum(1 for x in items if x.__class__.__name__ == "RepoFile")
    print(f"{video}\tremote_files={file_count}")
    if file_count == 0:
        missing.append(video)

if missing:
    raise SystemExit(f"Remote dataset missing dirs: {missing}")

print("PREFLIGHT_OK")
PY

echo "=== Step 2: low-concurrency staged download from HF ==="
python - <<'PY'
import os
import time
from huggingface_hub import snapshot_download

repo_id = os.environ["REPO_ID"]
revision = os.environ["REVISION"]
stage_root = os.environ["STAGE_ROOT"]

batches = [
    [
        "ytbv_38fe9b3ed1",
        "ytbv_6f96e91d81",
        "ytbv_73e4e5cc74",
        "ytbv_cb2e35d610",
        "ytbv_0503bf89c9",
    ],
    [
        "ytbv_aeba9ac967",
        "ytbv_45d9fc1357",
        "ytbv_045f00aed2",
        "ytbv_72cae058a5",
    ],
    [
        "ytbv_3b6e983b5b",
        "ytbv_04f21da964",
        "ytbv_6d12e30c48",
        "ytbv_804f6338a4",
    ],
    [
        "ytbv_a5ec5b0265",
        "ytbv_df365282c6",
        "davis_dogs-scale",
        "ytbv_da6c68708f",
    ],
]

for i, batch in enumerate(batches, 1):
    print(f"\n=== Download batch {i}/{len(batches)} ===")
    for v in batch:
        print("  ", v)

    allow_patterns = [f"{v}/*" for v in batch]

    for attempt in range(1, 4):
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
                local_dir=stage_root,
                allow_patterns=allow_patterns,
                max_workers=2,
                token=True,
            )
            print(f"Batch {i} OK on attempt {attempt}")
            break
        except Exception as e:
            print(f"Batch {i} failed on attempt {attempt}: {type(e).__name__}: {e}")
            if attempt == 3:
                raise
            sleep_sec = 30 * attempt
            print(f"Sleeping {sleep_sec}s before retry...")
            time.sleep(sleep_sec)

    if i != len(batches):
        print("Sleeping 20s before next batch to avoid HF burst I/O...")
        time.sleep(20)

print("\nDOWNLOAD_OK")
PY

echo "=== Step 3: verify staged files before overwrite ==="
python - <<'PY'
import os
from pathlib import Path
from collections import Counter
from PIL import Image

stage_root = Path(os.environ["STAGE_ROOT"])
videos = [
    "ytbv_38fe9b3ed1",
    "ytbv_6f96e91d81",
    "ytbv_73e4e5cc74",
    "ytbv_cb2e35d610",
    "ytbv_0503bf89c9",
    "ytbv_aeba9ac967",
    "ytbv_45d9fc1357",
    "ytbv_045f00aed2",
    "ytbv_72cae058a5",
    "ytbv_3b6e983b5b",
    "ytbv_04f21da964",
    "ytbv_6d12e30c48",
    "ytbv_804f6338a4",
    "ytbv_a5ec5b0265",
    "ytbv_df365282c6",
    "davis_dogs-scale",
    "ytbv_da6c68708f",
]

summary = Counter()
bad = []

for video in videos:
    vdir = stage_root / video
    if not vdir.exists():
        bad.append(("missing_dir", str(vdir)))
        continue

    for p in vdir.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue

        with open(p, "rb") as f:
            head = f.read(256)

        if head.startswith(b"version https://git-lfs.github.com/spec/v1"):
            summary["lfs_pointer"] += 1
            bad.append(("lfs_pointer", str(p)))
            continue

        try:
            with Image.open(p) as im:
                im.verify()
            summary["ok"] += 1
        except Exception as e:
            summary["invalid_image"] += 1
            bad.append((f"{type(e).__name__}", str(p)))

print("STAGE_SUMMARY", dict(summary))
print("STAGE_BAD_COUNT", len(bad))
for kind, path in bad[:50]:
    print(kind, path)

if bad:
    raise SystemExit("STAGE_VERIFY_FAILED")

print("STAGE_VERIFY_OK")
PY

echo "=== Step 4: overwrite the 17 affected dirs ==="
for d in \
  ytbv_38fe9b3ed1 \
  ytbv_6f96e91d81 \
  ytbv_73e4e5cc74 \
  ytbv_cb2e35d610 \
  ytbv_0503bf89c9 \
  ytbv_aeba9ac967 \
  ytbv_45d9fc1357 \
  ytbv_045f00aed2 \
  ytbv_72cae058a5 \
  ytbv_3b6e983b5b \
  ytbv_04f21da964 \
  ytbv_6d12e30c48 \
  ytbv_804f6338a4 \
  ytbv_a5ec5b0265 \
  ytbv_df365282c6 \
  davis_dogs-scale \
  ytbv_da6c68708f
do
  echo ">>> rsync $d"
  mkdir -p "$DATA_ROOT/$d"
  rsync -a "$STAGE_ROOT/$d/" "$DATA_ROOT/$d/"
done

echo "=== Step 5: full post-repair verification of the whole dataset ==="
python - <<'PY'
import os
import json
import time
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image

root = Path(os.environ["DATA_ROOT"])
paths = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}]

def check(path_str):
    path = Path(path_str)
    try:
        with open(path, "rb") as f:
            head = f.read(256)
        if head.startswith(b"version https://git-lfs.github.com/spec/v1"):
            return {"path": path_str, "status": "lfs_pointer", "detail": "git-lfs-pointer"}
        try:
            with Image.open(path) as im:
                im.verify()
            return {"path": path_str, "status": "ok", "detail": None}
        except Exception as e:
            return {"path": path_str, "status": "invalid_image", "detail": f"{type(e).__name__}: {e}"}
    except Exception as e:
        return {"path": path_str, "status": "unreadable_file", "detail": f"{type(e).__name__}: {e}"}

summary = Counter()
by_video = defaultdict(Counter)
bad = []

start = time.time()
workers = min(8, os.cpu_count() or 4)
processed = 0

with ProcessPoolExecutor(max_workers=workers) as ex:
    futures = [ex.submit(check, str(p)) for p in paths]
    for fut in as_completed(futures):
        res = fut.result()
        processed += 1
        rel = str(Path(res["path"]).relative_to(root))
        video = rel.split("/")[0]
        summary[res["status"]] += 1
        by_video[video][res["status"]] += 1
        if res["status"] != "ok":
            bad.append({
                "relative_path": rel,
                "status": res["status"],
                "detail": res["detail"],
            })
        if processed % 50000 == 0 or processed == len(paths):
            elapsed = time.time() - start
            rate = processed / elapsed if elapsed > 0 else 0
            print(f"VERIFY_PROGRESS {processed}/{len(paths)} rate={rate:.1f}/s", flush=True)

videos_with_bad = []
for video, counts in by_video.items():
    bad_total = counts["lfs_pointer"] + counts["invalid_image"] + counts["unreadable_file"]
    if bad_total:
        videos_with_bad.append({
            "video": video,
            "bad_total": bad_total,
            "lfs_pointer": counts["lfs_pointer"],
            "invalid_image": counts["invalid_image"],
            "unreadable_file": counts["unreadable_file"],
            "ok": counts["ok"],
        })
videos_with_bad.sort(key=lambda x: (-x["bad_total"], x["video"]))

report = {
    "root": str(root),
    "total_images": len(paths),
    "summary": dict(summary),
    "bad_count": len(bad),
    "videos_with_bad_count": len(videos_with_bad),
    "videos_with_bad": videos_with_bad,
    "bad_files": bad,
}

report_json = Path("/tmp/dpo_dataset_post_repair_verify.json")
report_txt = Path("/tmp/dpo_dataset_post_repair_bad_files.txt")
report_json.write_text(json.dumps(report, indent=2))
with report_txt.open("w") as f:
    for item in bad:
        f.write(f"{item['status']}\t{item['relative_path']}\t{item['detail']}\n")

print("\n=== FINAL SUMMARY ===")
print(json.dumps({
    "total_images": len(paths),
    "ok": summary["ok"],
    "lfs_pointer": summary["lfs_pointer"],
    "invalid_image": summary["invalid_image"],
    "unreadable_file": summary["unreadable_file"],
    "bad_count": len(bad),
    "videos_with_bad_count": len(videos_with_bad),
    "report_json": str(report_json),
    "report_txt": str(report_txt),
}, indent=2))

if bad:
    print("\n=== REMAINING BAD FILES (first 100) ===")
    for item in bad[:100]:
        print(item["status"], item["relative_path"], item["detail"])
    raise SystemExit("FINAL_STATUS=FAILED")
else:
    print("FINAL_STATUS=OK")

PY
