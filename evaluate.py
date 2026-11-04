import gzip, json, sys
from collections import defaultdict


def load(path):
    d = json.load(gzip.open(path, "rt") if path.endswith(".gz") else open(path))
    names = {c["id"]: c["name"] for c in d["categories"]}
    ims = {i["id"]: (i["video"], i["frame"]) for i in d["images"]}
    out = defaultdict(list)
    for i in ims.values():
        out[i]
    for a in d["annotations"]:
        x, y, w, h = a["bbox"]
        out[ims[a["image_id"]]].append((names[a["category_id"]], (x, y, x + w, y + h)))
    return out


def iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    i = w * h
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def evaluate(pred, ref):
    tp = fp = fn = 0
    ious, matched = [], []
    for k, rb in ref.items():
        pb = pred.get(k, [])
        rt, pt = {c for c, _ in rb}, {c for c, _ in pb}
        tp, fp, fn = tp + len(rt & pt), fp + len(pt - rt), fn + len(rt - pt)
        for c in rt:
            r = [b for t, b in rb if t == c]
            p = [b for t, b in pb if t == c]
            pairs = sorted(((iou(x, y), i, j) for i, x in enumerate(r) for j, y in enumerate(p)), reverse=True)
            ur, up = set(), set()
            for v, i, j in pairs:
                if i not in ur and j not in up:
                    ur.add(i); up.add(j); ious.append(v); matched.append(v)
            ious += [0.0] * (len(r) - len(ur))
    mean = lambda v: sum(v) / len(v) if v else 0.0
    return {"set_f1": 200 * tp / (2 * tp + fp + fn), "miou": mean(ious), "matched_miou": mean(matched)}


if __name__ == "__main__":
    pred, ref = load(sys.argv[1]), load(sys.argv[2])
    for k, v in evaluate(pred, {k: v for k, v in ref.items()}).items():
        print(f"{k:14s} {v:.3f}")
