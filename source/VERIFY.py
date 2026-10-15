#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,os
root=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()
receipt=json.loads((root/'INSTALLATION.json').read_text())
assert root==Path(receipt['installed_root']), 'Use INSTALL.py to relocate an installation'
for name,item in receipt['files'].items():
    p=root/'payload'/name
    assert (p.is_symlink() and os.readlink(p)==item['link']) if 'link' in item else (p.is_file() and not p.is_symlink() and p.stat().st_size==item['bytes'] and sha(p)==item['sha256']),name
for p,h in receipt['foundation']['files'].items():assert sha(p)==h,p
model=Path(receipt['model']);proof=json.loads((root/'MODEL.json').read_text())
assert sha(model/'conversion-manifest.json')==proof['conversion_manifest_sha256']
for name,item in proof['files'].items():
    p=model/name
    assert p.is_file() and p.stat().st_size==item['bytes'],name
    if not name.endswith('.safetensors'):assert sha(p)==item['sha256'],name
print(json.dumps({'verified':True,'source_pins':94,'installed_files':len(receipt['files']),'model':str(model),'weights_rehashed':False}))
