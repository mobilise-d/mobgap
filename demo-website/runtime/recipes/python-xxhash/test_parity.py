"""Independent native golden vectors exercise real WASM extension hashing."""

import json
from pathlib import Path

import xxhash

payloads = {
    "empty": b"",
    "abc": b"abc",
    "all_bytes": bytes(range(256)),
    "repeated": b"a" * 100000,
    "large": bytes(range(251)) * 4000,
}
vectors = json.loads(Path(__file__).with_name("parity_vectors.json").read_text())
for v in vectors:
    factory = getattr(xxhash, v["algorithm"])
    data = payloads[v["payload"]]
    direct = factory(data, seed=v["seed"])
    assert direct.hexdigest() == v["hex"], v
    assert str(direct.intdigest()) == v["int"], v
    assert direct.digest() == bytes.fromhex(v["hex"]), v
    incremental = factory(seed=v["seed"])
    middle = len(data) // 2
    incremental.update(memoryview(data)[:middle])
    copied = incremental.copy()
    incremental.update(bytearray(data[middle:]))
    copied.update(data[middle:])
    assert incremental.hexdigest() == copied.hexdigest() == v["hex"], v
    copied.reset()
    copied.update(data)
    assert copied.hexdigest() == v["hex"], v
print(
    {
        "xxhash_version": xxhash.VERSION,
        "native_golden_vectors": len(vectors),
        "checks": "direct/seed/int/digest/incremental/copy/reset passed",
    }
)
