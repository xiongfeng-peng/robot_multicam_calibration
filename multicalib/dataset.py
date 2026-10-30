"""Sample persistence with both observations and camera images for audit/reprocessing."""
import json
from pathlib import Path
import numpy as np
import cv2


def samples_dir(data_dir, stage):
    if stage not in ("wrist", "cam1", "cam2"):
        raise ValueError(stage)
    return Path(data_dir) / "samples" / stage


def save_sample(data_dir, stage, poses, images, metadata=None):
    root = samples_dir(data_dir, stage)
    root.mkdir(parents=True, exist_ok=True)
    ids = [int(p.name.split('_')[-1]) for p in root.glob('sample_*') if p.is_dir() and p.name.split('_')[-1].isdigit()]
    i = max(ids, default=-1) + 1
    directory = root / f"sample_{i:04d}"
    directory.mkdir(parents=True, exist_ok=False)
    j = {"stage": stage, "sequence": i, "poses": {}, "metadata": metadata or {}}
    for key, T in poses.items():
        j["poses"][key] = np.asarray(T, float).reshape(4, 4).tolist()
    for key, image in images.items():
        if image is not None:
            if not cv2.imwrite(str(directory / f"{key}.png"), image):
                raise IOError(f"Failed saving image {key}")
    (directory / 'observation.json').write_text(json.dumps(j, indent=2), encoding='utf-8')
    return directory


def load_samples(data_dir, stage):
    out = []
    for path in sorted(samples_dir(data_dir, stage).glob("sample_*/observation.json")):
        j = json.loads(path.read_text(encoding='utf-8'))
        out.append({"path": str(path.parent), "poses": {k: np.asarray(T, float) for k, T in j['poses'].items()},
                    "metadata": j.get('metadata', {})})
    return out


def save_transform(result_dir, name, T, meta):
    directory = Path(result_dir); directory.mkdir(parents=True, exist_ok=True)
    T = np.asarray(T, float).reshape(4, 4)
    np.save(directory / f"{name}.npy", T)
    with (directory / f"{name}.json").open("w", encoding="utf-8") as f:
        json.dump({"name": name, "convention": "T_target_source (point_target = T @ point_source)",
                   "units": "meters", "matrix": T.tolist(), "metrics": meta}, f, indent=2)


def read_transform(result_dir, name):
    path = Path(result_dir) / f"{name}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run wrist solve first.")
    return np.load(path)
