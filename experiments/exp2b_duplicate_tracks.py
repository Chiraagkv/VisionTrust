"""
Experiment 2b — Are the "duplicates" found in unmodified GTSRB real?

GTSRB stores ~30 consecutive video frames of each physical sign (a "track";
file names are TTTTT_FFFFF.ppm). This checks what share of the duplicate pairs
the pHash detector reports on the clean subset are frames of the same track.

Run:  python experiments/exp2b_duplicate_tracks.py   (GTSRB only)
"""
from pathlib import Path

import torchvision.datasets as tvd

from common import RAW_DIR, platform_dataset, save_json
from assurance.dataset_scanner import detect_duplicates


def main():
    base = platform_dataset("GTSRB", "train", 8600)
    raw = tvd.GTSRB(str(RAW_DIR), split="train", download=False)
    track = {}
    for r in base.records:
        p = Path(raw._samples[r.extra["source_index"]][0])
        track[r.filename] = (p.parent.name, p.name.split("_")[0])     # (class folder, track id)
    out = {}
    for thr in (12, 24, 32):
        groups, dups, _ = detect_duplicates(base.records, threshold=thr)
        pairs = [(g.representative, d) for g in groups for d in g.duplicates]
        same = sum(track[a] == track[b] for a, b in pairs)
        out[thr] = {"duplicates": len(dups), "pairs": len(pairs), "same_track": same,
                    "share_same_track": same / max(len(pairs), 1)}
        print(thr, out[thr])
    save_json("exp2b_gtsrb_duplicate_tracks", out)


if __name__ == "__main__":
    main()
