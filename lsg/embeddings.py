"""
Sentence embedding with all-MiniLM-L6-v2, cached to disk.

On Python 3.14 / arm64 macOS, sentence-transformers' tokenizer creates a
loky semaphore that conflicts with XGBoost's OpenMP runtime, causing a
segfault when both are used in the same process.  We work around this by
running the encode step in a subprocess that exits before any ML code runs.
"""
import hashlib
import os
import subprocess
import sys
import tempfile

import numpy as np

CACHE_DIR = ".emb_cache"


def encode(texts: list[str], batch_size: int = 256) -> np.ndarray:
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = hashlib.md5("".join(texts).encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"{key}.npy")
    if os.path.exists(cache_path):
        return np.load(cache_path)

    # Write texts to a temp file so the subprocess can read them
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("\n".join(t.replace("\n", " ") for t in texts))
        txt_path = f.name

    script = f"""
import numpy as np
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
with open({repr(txt_path)}) as f:
    texts = f.read().splitlines()
embs = model.encode(texts, batch_size={batch_size}, show_progress_bar=True, convert_to_numpy=True)
np.save({repr(cache_path)}, embs)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=False,   # let progress bars print to terminal
    )
    os.unlink(txt_path)

    if result.returncode != 0:
        raise RuntimeError(f"Embedding subprocess failed with code {result.returncode}")

    return np.load(cache_path)
