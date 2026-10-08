import hashlib
import json
import sys
import tarfile
from pathlib import Path

artifact = Path(sys.argv[1])
with tarfile.open(artifact) as package:
    index = json.load(package.extractfile("info/index.json"))
    assert index["version"] == "1.9.0"
    assert index["subdir"] == "emscripten-wasm32"
    extensions = [m for m in package.getmembers() if m.name.endswith(".so")]
    assert len(extensions) == 4
    for module in extensions:
        assert package.extractfile(module).read(8) == b"\0asm\x01\0\0\0", module.name
    record = {
        "index": index,
        "extensions": {m.name: m.size for m in extensions},
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    }
print(json.dumps(record, indent=2))
