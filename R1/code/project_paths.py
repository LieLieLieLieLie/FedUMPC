"""Explicit paths and relocation-aware provenance for the R1 experiments."""
from pathlib import Path
import hashlib
import json
import os

R1_ROOT = Path(__file__).resolve().parent
_MAP = json.loads((R1_ROOT / 'scripts/relocation_map.json').read_text(encoding='utf-8'))

def artifact_path(relative):
    """Resolve an original result name below the grouped artifacts tree."""
    parts = Path(relative).parts
    mapped = Path(_MAP['data'][parts[0]]).joinpath(*parts[1:])
    override = os.environ.get('FEDUMPC_ARTIFACTS_ROOT')
    return (Path(override).resolve() / mapped.relative_to('artifacts')) if override else R1_ROOT / mapped

def source_file(name):
    return R1_ROOT / _MAP['scripts'].get(str(name), str(name))

def source_digest(name):
    return hashlib.sha256(source_file(name).read_bytes()).hexdigest()

def _canonical_hash(value):
    """Recognize only hashes of precisely recorded path-only refactors."""
    record = R1_ROOT / 'scripts/relocation_verification.json'
    if not record.exists():
        return value
    changes = json.loads(record.read_text(encoding='utf-8'))['source_changes']
    return next((r['before_sha256'] for r in changes if r['after_sha256'] == value), value)

def verify_source_hash(name, expected):
    actual = source_digest(name)
    if actual != expected and _canonical_hash(actual) != _canonical_hash(expected):
        raise RuntimeError(f'Source differs from the recorded implementation: {name}')

def check_protocol(recorded, current):
    """Check scientific settings; retain original runtime/source provenance.

    Paths may change and installed runtime versions are recorded per invocation.
    Only hashes explicitly present in the relocation audit are equivalent.
    """
    def normalize(value):
        if isinstance(value, dict):
            return {k: normalize(v) for k,v in value.items()
                    if k not in {'python','numpy','torch','environment'}}
        if isinstance(value, list):
            return [normalize(v) for v in value]
        return _canonical_hash(value) if isinstance(value,str) and len(value)==64 else value
    if normalize(recorded) != normalize(current):
        raise RuntimeError('Experimental settings changed; use a separate result directory')
