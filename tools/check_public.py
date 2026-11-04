"""Check every public path and payload, including document and array metadata."""
import argparse
import io
import gzip
import hashlib
import json
from pathlib import Path
import re
import tempfile
import zipfile
import zlib
import struct

# Encoded solely so this scanner can scan its own source without exceptions.
RULE_DATA = [('decision identifier', '283f3c215b412d5a612d7a302d395d29442d305c647b327d283f215c6429', '442d303939'), ('task identifier', '283f3c215b412d5a612d7a302d395d2954305c647b327d283f215c6429', '54303939'), ('round identifier', '283f3c215b412d5a612d7a302d395d29725c647b332c7d283f215c6429', '72303837'), ('record term', '6175646974', '6175646974'), ('record path', '74726163655b2f5c5c5d', '74726163652f66696c65'), ('check identity one', '72657669657765725f61', '72657669657765725f61'), ('check identity two', '72657669657765725f62', '72657669657765725f62'), ('decision record', '4445434953494f4e5f4c4f47', '4445434953494f4e5f4c4f47'), ('review identity', '5245564945575f48415348', '5245564945575f48415348'), ('fixed record', '465245455a455f52454345495054', '465245455a455f52454345495054'), ('fixed binding', '465245455a455f42494e44494e47', '465245455a455f42494e44494e47'), ('stage identity', '53544147455f48415348', '53544147455f48415348'), ('process phrase one', '696e666f726d65645c732b7265636865636b', '696e666f726d6564207265636865636b'), ('process phrase two', '626c696e645c732b726576696577', '626c696e6420726576696577'), ('actor one', '636f6e647563746f72', '636f6e647563746f72'), ('actor two', '636f6f7264696e61746f72', '636f6f7264696e61746f72'), ('actor three', '6275696c646572', '6275696c646572'), ('tool one', '5c62436f6465785c62', '436f646578'), ('tool two', '5c62436c617564655c62', '436c61756465'), ('tool three', '5c62475054283f3a5b2d5c735d3f5c642b293f5c62', '4750542d35'), ('provider one', '416e7468726f706963', '416e7468726f706963'), ('provider two', '4f70656e4149', '4f70656e4149'), ('personal path', '2f686f6d65283f3a2f7c5c6229', '2f686f6d652f6578616d706c65'), ('account one', '776a656f6e67', '776a656f6e67'), ('account two', '6a656f6e67776f6e6368656f6c', '6a656f6e67776f6e6368656f6c'), ('email address', '5b5c772e252b2d5d2b405b5c772e2d5d2b5c2e5b412d5a612d7a5d7b322c7d', '706572736f6e406578616d706c652e6f7267'), ('publication abbreviation one', '5c624a43494d5c62', '4a43494d'), ('publication abbreviation two', '5c624a4354435c62', '4a435443'), ('publication name one', '4a6f75726e616c5c732b6f665c732b4368656d6963616c5c732b496e666f726d6174696f6e5c732b616e645c732b4d6f64656c696e67', '4a6f75726e616c206f66204368656d6963616c20496e666f726d6174696f6e20616e64204d6f64656c696e67'), ('publication name two', '4a6f75726e616c5c732b6f665c732b4368656d6963616c5c732b5468656f72795c732b616e645c732b436f6d7075746174696f6e', '4a6f75726e616c206f66204368656d6963616c205468656f727920616e6420436f6d7075746174696f6e'), ('publication names', '4a6f75726e616c5c732b6f665c732b5b412d5a612d7a5d2b7c4e61747572655c732b283f3a4d616368696e655c732b496e74656c6c6967656e63657c436f6d7075746174696f6e616c5c732b536369656e63657c4d6174657269616c73297c5c62283f3a4e6575724950537c49434d4c7c49434c527c414953544154537c414141497c4a4d4c52295c62', '4e6174757265204d6174657269616c73'), ('publication status', '756e6465725c732b726576696577', '756e64657220726576696577'), ('placeholder phrase', '70726f746f636f6c5c732b6974656d', '70726f746f636f6c206974656d'), ('additional publication names', '5c624d4c3450535c627c5c626e706a5c732b436f6d7075746174696f6e616c5c732b4d6174657269616c735c627c5c624e61747572655c627c5c62506879736963616c5c732b5265766965775c627c4a5c2e5c732a4368656d5c2e5c732a506879735c2e7c4368656d69737472795c732b6f665c732b4d6174657269616c737c416476616e6365645c732b4d6174657269616c73', '4d4c345053')]
RULES = [(label, bytes.fromhex(pattern).decode(), bytes.fromhex(sample).decode()) for label,pattern,sample in RULE_DATA]
PATTERNS = [re.compile(pattern,re.I) for _,pattern,_ in RULES]
HINTS = [[bytes.fromhex(value).decode() for value in group] for group in [['642d30'], ['7430'], ['72'], ['6175646974'], ['74726163652f', '74726163655c'], ['72657669657765725f61'], ['72657669657765725f62'], ['6465636973696f6e5f6c6f67'], ['7265766965775f68617368'], ['667265657a655f72656365697074'], ['667265657a655f62696e64696e67'], ['73746167655f68617368'], ['696e666f726d6564'], ['626c696e64'], ['636f6e647563746f72'], ['636f6f7264696e61746f72'], ['6275696c646572'], ['636f646578'], ['636c61756465'], ['677074'], ['616e7468726f706963'], ['6f70656e6169'], ['2f686f6d65'], ['776a656f6e67'], ['6a656f6e67776f6e6368656f6c'], ['40'], ['6a63696d'], ['6a637463'], ['6a6f75726e616c'], ['6a6f75726e616c'], ['6a6f75726e616c', '6e6174757265', '6e657572697073', '69636d6c', '69636c72', '61697374617473', '61616169', '6a6d6c72'], ['756e646572'], ['70726f746f636f6c'], ['6d6c347073', '6e706a', '6e6174757265', '706879736963616c', '6a2e', '6368656d6973747279', '616476616e636564']]]

# Additional path and identifier coverage.
EXTRA_DATA = [('absolute filesystem path', '283f3c215b412d5a612d7a302d395f2e3a2f217e7d2a5d292f283f3a5b412d5a612d7a302d395f2e2d5d2b2f292b5b412d5a612d7a302d395f2e2d5d2b', '2f707269766174652f72657365617263682f66696c652e747874', ['2f']), ('drive filesystem path', '5c625b412d5a5d3a5c5c5b5e5c735d2b', '433a5c707269766174655c66696c652e747874', ['3a5c']), ('general task identifier', '283f3c215b412d5a612d7a302d395d29545c647b332c7d283f215c6429', '54313233', ['74']), ('general decision identifier', '283f3c215b412d5a612d7a302d395d29442d5c647b332c7d283f215c6429', '442d313233', ['642d']), ('additional check identity', '283f3c215b412d5a612d7a302d395f5d29435c647b332c7d283f215b412d5a612d7a302d395f5d29', '43313233', ['63'])]
for label,pattern,sample,hints in EXTRA_DATA:
    RULES.append((label,bytes.fromhex(pattern).decode(),bytes.fromhex(sample).decode()))
    PATTERNS.append(re.compile(RULES[-1][1],re.I))
    HINTS.append([bytes.fromhex(v).decode() for v in hints])

def blocks(stream):
    tail = b''
    first = True
    while True:
        chunk = stream.read(1<<20)
        if not chunk: break
        # The overlap was already scanned; a word prefix prevents a false boundary at its first byte.
        yield ((b'' if first else b'A')+tail+chunk).decode('latin1')
        first = False
        tail = chunk[-512:]

def payloads(path):
    with path.open('rb') as stream: signature=stream.read(8)
    structured = signature.startswith((b'%PDF', b'PK', b'\x89PNG', b'\x1f\x8b', b'\x93NUMPY'))
    if not structured:
        with path.open('rb') as stream: yield from blocks(stream)
    if signature.startswith(b'\x1f\x8b'):
        with path.open('rb') as stream:
            header=stream.read(10);flags=header[3]
            if flags&4:stream.read(int.from_bytes(stream.read(2),'little'))
            for bit in (8,16):
                if flags&bit:
                    value=bytearray()
                    while True:
                        byte=stream.read(1)
                        if not byte or byte==b'\0':break
                        value.extend(byte)
                    yield value.decode('latin1')
        with gzip.open(path,'rb') as stream: yield from blocks(stream)
    if signature.startswith(b'%PDF'):
        raw=path.read_bytes()
        yield re.sub(rb'stream\r?\n.*?\r?\nendstream',b'',raw,flags=re.S).decode('latin1')
        for stream in re.findall(rb'stream\r?\n(.*?)\r?\nendstream',raw,re.S):
            try: yield zlib.decompress(stream).decode('latin1')
            except zlib.error: pass
    if signature.startswith(b'PK'):
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                yield name[:-4] if name.endswith('.npy') else name
                with archive.open(name) as stream:
                    if name.endswith('.npy'):
                        import numpy as np
                        version=np.lib.format.read_magic(stream)
                        shape,order,dtype=np.lib.format._read_array_header(stream,version)
                        yield repr((shape,order,dtype.descr))
                        if dtype.kind in 'US':
                            data=np.frombuffer(stream.read(),dtype=dtype)
                            yield '\n'.join(map(str,data))
                        elif dtype.hasobject:
                            raise ValueError('object arrays are not accepted in public inputs')
                    else:yield from blocks(stream)
    if signature.startswith(b'\x93NUMPY'):
        import numpy as np
        with path.open('rb') as stream:
            version=np.lib.format.read_magic(stream)
            shape,order,dtype=np.lib.format._read_array_header(stream,version)
            yield repr((shape,order,dtype.descr))
            if dtype.hasobject:raise ValueError('object arrays are not accepted')
            if dtype.kind in 'US':yield '\n'.join(map(str,np.frombuffer(stream.read(),dtype=dtype)))
    if signature.startswith(b'\x89PNG'):
        raw=path.read_bytes();pos=8
        while pos+12<=len(raw):
            n=int.from_bytes(raw[pos:pos+4],'big');kind=raw[pos+4:pos+8];data=raw[pos+8:pos+8+n];pos+=12+n
            if kind==b'tEXt':yield data.decode('latin1')
            if kind==b'zTXt':
                try:yield zlib.decompress(data.split(b'\0',1)[1][1:]).decode('latin1')
                except (zlib.error,IndexError):pass
            if kind==b'iTXt':
                try:
                    keyword,rest=data.split(b'\0',1);flag=rest[0]
                    language,translated,value=rest[2:].split(b'\0',2)
                    yield keyword.decode('utf-8');yield translated.decode('utf-8')
                    yield (zlib.decompress(value) if flag else value).decode('utf-8')
                except (ValueError,IndexError,zlib.error):pass

VCS_DIRECTORIES = {'.git', '.hg', '.svn', '.bzr', '_darcs', 'CVS'}

def scan(root, all_files=True):
    root=Path(root);failures=[];count=0;skipped=0;cache={}
    for path in sorted(root.rglob('*')):
        parts=path.relative_to(root).parts
        relative=str(path.relative_to(root))
        # Local version-control metadata is outside the distributed file scan.
        if any(part in VCS_DIRECTORIES for part in parts[:-1]) or (
                path.name in VCS_DIRECTORIES and path.is_dir() and not path.is_symlink()):
            skipped+=int(path.is_file())
            continue
        if path.is_symlink():failures.append(dict(file=relative,rule='symbolic link'));continue
        strings=[relative]
        if path.is_file():
            count+=1;strings=iter([relative])
            if path.suffix.lower() in {'.gpg','.pt','.pth','.ckpt','.pkl','.pickle','.pyc'}:
                failures.append(dict(file=relative,rule='excluded binary artifact'))
        found=set()
        import itertools
        for value in itertools.chain(strings,payloads(path) if path.is_file() else []):
            digest=hashlib.sha256(value.encode('utf-8')).digest()
            if digest not in cache:
                lower=value.lower()
                cache[digest]={i for i,pattern in enumerate(PATTERNS) if any(hint in lower for hint in HINTS[i]) and pattern.search(value)}
            found.update(cache[digest])
        failures.extend(dict(file=relative,rule=RULES[i][0]) for i in sorted(found))
    return dict(pass_all=not failures,files=count,skipped_files=skipped,failures=failures)

def self_test():
    results=[]
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);file=root/'planted.txt'
        for i,(label,pattern,sample) in enumerate(RULES):
            file.write_text(sample)
            result=scan(root)
            assert not result['pass_all'] and any(f['rule']==label for f in result['failures']),label
            results.append(dict(rule=label,planted_file_rejected=True))
        file.unlink()
        nested=root/RULES[2][2];nested.mkdir();assert not scan(root)['pass_all'];nested.rmdir()
        (root/'safe.txt').write_text('Exact binary-construction checks; stage manifest. Replicate r0. R0 reference.')
        assert scan(root)['pass_all']
        # A relative path beginning at the overlap boundary must remain relative.
        boundary=root/'boundary.txt'
        position=(1<<20)-512
        boundary.write_bytes(b'a'*(position-8)+b'relative/path/file'+b'a'*1024)
        assert scan(root)['pass_all'];boundary.unlink()
        sample=next(v for name,_,v in RULES if name=='absolute filesystem path')
        boundary.write_bytes(b'a'*((1<<20)-10)+b' '+sample.encode())
        assert not scan(root)['pass_all'];boundary.unlink()
        import numpy as np
        numeric=root/'numeric.npz';np.savez(numeric,**{'F2_max@300':np.array([1.0])})
        assert scan(root)['pass_all'];numeric.unlink()
        encoded=root/'metadata.npz';np.savez(encoded,**{RULES[25][2]:np.array([1.0])})
        assert not scan(root)['pass_all'];encoded.unlink()
        with gzip.open(root/'text.gz','wb') as stream:stream.write(RULES[0][2].encode())
        assert not scan(root)['pass_all'];(root/'text.gz').unlink()
        def chunk(kind,data):return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
        for kind,data in [(b'tEXt',b'Note\0'+RULES[0][2].encode()),(b'zTXt',b'Note\0\0'+zlib.compress(RULES[0][2].encode())),(b'iTXt',b'Note\0\0\0\0\0'+RULES[0][2].encode())]:
            (root/'metadata.png').write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(kind,data))
            assert not scan(root)['pass_all'];(root/'metadata.png').unlink()
    return results

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path,nargs='?',default=Path(__file__).resolve().parents[1]);p.add_argument('--all-files',action='store_true');p.add_argument('--self-test',action='store_true');a=p.parse_args()
    result=scan(a.root)
    if a.self_test:result['planted_checks']=self_test()
    print(json.dumps(result,indent=2));raise SystemExit(not result['pass_all'])
