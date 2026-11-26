"""Client side of the sketch -> label backend.

Request (JSON):
    {
      "format": "stroke3",
      "points": [[dx, dy, pen_up], ...],   # normalized, see preprocess.py
      "length": T,
      "top_k": 5
    }

Response (JSON):
    {
      "predictions": [{"label": "cat", "score": 0.83}, ...],   # best first
      "neighbors": [                                            # optional
        {"id": "cat/full/train/12", "label": "cat", "score": 0.91,
         "points": [[dx, dy, pen_up], ...]}
      ]
    }

`neighbors` is for the retrieval model: training sketches closest to the
query in feature space. The frontend shows them when they are present.
"""

import json
import urllib.request


def build_request(points, top_k):
    return {"format": "stroke3", "points": [[float(dx), float(dy), int(pen)] for dx, dy, pen in points],
            "length": len(points), "top_k": top_k}


class HttpBackend:
    name = "http"

    def __init__(self, url, timeout=10.0):
        self.url = url
        self.timeout = timeout

    def query(self, request):
        http_request = urllib.request.Request(
            self.url, data=json.dumps(request).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(http_request, timeout=self.timeout) as response:
            return json.load(response)


class MockBackend:
    """Placeholder until the real backend exists. Its answers are meaningless."""

    name = "mock"
    LABELS = ["cat", "dog", "house", "tree", "sun", "fish", "flower", "bicycle", "airplane", "apple"]

    def query(self, request):
        count = min(request["top_k"], len(self.LABELS))
        first = request["length"] % len(self.LABELS)
        labels = [self.LABELS[(first + i) % len(self.LABELS)] for i in range(count)]
        weights = [0.5 ** (i + 1) for i in range(count)]
        predictions = [{"label": label, "score": weight / sum(weights)} for label, weight in zip(labels, weights)]
        # Echo the query back so the neighbor panels can be exercised too.
        neighbors = [{"id": f"mock/{i}", "label": p["label"], "score": p["score"], "points": request["points"]}
                     for i, p in enumerate(predictions[:3])]
        return {"predictions": predictions, "neighbors": neighbors}


def make_backend(config):
    kind = config.get("type", "mock")
    if kind == "mock":
        return MockBackend()
    if kind == "http":
        return HttpBackend(config["url"], config.get("timeout", 10.0))
    raise ValueError(f"Unknown backend type: {kind}")
