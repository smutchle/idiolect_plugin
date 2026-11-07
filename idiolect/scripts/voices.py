"""List saved voices: genres, corpus size, samples, calibration, sources.

    python voices.py [--json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import idiolect_home, read_json  # noqa: E402


def list_voices() -> list[dict]:
    root = idiolect_home() / "voices"
    out = []
    if not root.exists():
        return out
    for vdir in sorted(p for p in root.iterdir() if p.is_dir()):
        for gdir in sorted(p for p in vdir.iterdir() if p.is_dir()):
            pf = gdir / "profile.json"
            if not pf.exists():
                continue
            prof = read_json(pf)
            cal = read_json(gdir / "calibration.json") if (gdir / "calibration.json").exists() else None
            b = (cal or {}).get("bands", {}).get("1000", {}).get("composite", {})
            out.append({"voice": vdir.name, "genre": gdir.name, "words": prof["corpus"]["words"],
                        "samples": prof["corpus"]["samples"], "files": len(prof["meta"].get("sources", [])),
                        "created": prof["meta"].get("created"), "calibrated": cal is not None,
                        "band_1000w": [round(b["lo"], 2), round(b["hi"], 2)] if b else None,
                        "warnings": len(prof.get("warnings", [])), "path": str(gdir)})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    vs = list_voices()
    if a.json:
        print(json.dumps(vs, indent=1))
        return
    if not vs:
        print(f"No voices yet in {idiolect_home() / 'voices'}. Create one with /idiolect:voice-profile.")
        return
    print("| voice | genre | words | samples | files | calibrated | band @1000w | created |")
    print("|---|---|---|---|---|---|---|---|")
    for v in vs:
        band = f"{v['band_1000w'][0]}-{v['band_1000w'][1]}" if v["band_1000w"] else "-"
        flag = " (low: <20k)" if v["words"] < 20000 else ""
        print(f"| {v['voice']} | {v['genre']} | {v['words']:,}{flag} | {v['samples']} | {v['files']} | "
              f"{'yes' if v['calibrated'] else 'no'} | {band} | {(v['created'] or '')[:10]} |")


if __name__ == "__main__":
    main()
