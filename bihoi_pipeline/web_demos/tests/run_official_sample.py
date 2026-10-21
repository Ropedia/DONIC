"""Run an available upstream sample through the actual HTTP upload/job API."""
import argparse
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("wapr", "mose3"))
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    args = parser.parse_args()

    def request(path, body=None, headers=None):
        with urlopen(Request(args.url + path, body, headers or {}), timeout=120) as response:
            return json.load(response)

    def upload(path):
        return request("/api/upload", path.read_bytes(), {
            "Content-Type": "application/octet-stream", "X-Filename":path.name})["token"]

    if args.model == "wapr":
        sample = ROOT / "runtime/cache/wapr/cache/samples/pose_lmo"
        meta = json.loads((sample / "meta.json").read_text())
        body = dict(app="wapr", device="cuda:0", region="mask", backend="torch",
                    depth_scale=1, mesh_scale=1,
                    **{k:meta[k] for k in ("fx", "fy", "cx", "cy", "diameter_m")})
        for key, name in {"rgb":"rgb.png", "depth":"depth.npy", "mesh":"model.ply", "mask":"mask.png"}.items():
            body[key] = upload(sample / name)
    else:
        body = dict(app="mose3", device="cuda:0", video=upload(ROOT / "third_party/MoSE3/examples/spin.mp4"),
                    num_frames=8, resolution=224, num_ref_pts=1024, query_idx=0, start_frame=0,
                    frame_interval=0, stride=48, fps=8, se3_sigma_px=8)
    job = request("/api/jobs", json.dumps(body).encode(), {"Content-Type":"application/json"})
    report = ROOT / "runtime" / (args.model + "_sample_job.json")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(job))
    print(f"Submitted {args.model}: {job['id']}", flush=True)
    seen = 0
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        job = request("/api/jobs/" + job["id"])
        print(job["log"][seen:], end="", flush=True)
        seen = len(job["log"])
        if job["status"] not in ("queued", "running"):
            report.write_text(json.dumps(job, ensure_ascii=False, indent=2))
            print("\nResult:", json.dumps(job["result"], ensure_ascii=False), flush=True)
            print("Artifacts:", job["artifacts"], flush=True)
            if job["status"] != "succeeded":
                raise SystemExit("Sample failed: " + job["status"])
            return
        time.sleep(2)
    raise SystemExit("Timed out; job remains available from the web interface")


if __name__ == "__main__":
    main()
