"""Download publisher-owned inputs and require the recorded SHA-256."""
import hashlib
import json
from pathlib import Path
import urllib.request
ROOT=Path(__file__).resolve().parents[1]
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()
def main():
    for spec in json.loads((ROOT/'data_sources.json').read_text())['files']:
        path=ROOT/spec['path'];path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            if digest(path)!=spec['sha256']:raise ValueError('existing input hash mismatch: '+spec['path'])
            print('Verified',spec['path']);continue
        partial=path.with_name(path.name+'.partial')
        urllib.request.urlretrieve(spec['url'],partial)
        if digest(partial)!=spec['sha256']:raise ValueError('download hash mismatch: '+spec['path'])
        partial.rename(path);print('Verified',spec['path'])
if __name__=='__main__':main()
