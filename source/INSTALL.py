#!/usr/bin/env python3
"""Install the current94 RC into a new directory. Never changes services/profiles."""
import argparse, hashlib, json, os, pathlib, re, shutil, sys

ROOT=pathlib.Path(__file__).resolve().parent
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()
def read(p):return json.loads(p.read_text())
def save(p,obj):p.write_text(json.dumps(obj,indent=2)+'\n')
def rewrite(p,text):
    if p.read_text()==text:return
    mode=p.stat().st_mode
    p.chmod(mode|0o200)
    try:p.write_text(text)
    finally:p.chmod(mode)
def inventory(root):
    return {str(p.relative_to(root)):({'link':os.readlink(p)} if p.is_symlink() else {'sha256':sha(p),'bytes':p.stat().st_size})
        for p in sorted(root.rglob('*')) if p.is_file() or p.is_symlink()}
def verify(root,files):
    for name,item in files.items():
        rel=pathlib.PurePosixPath(name)
        if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe inventory path: '+name)
        p=root/name
        if 'link' in item:
            if not p.is_symlink() or os.readlink(p)!=item['link']:raise ValueError('Changed link: '+name)
        elif not p.is_file() or p.is_symlink() or p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:
            raise ValueError('Checksum mismatch: '+name)
def model_check(model,proof,full):
    if sha(model/'conversion-manifest.json')!=proof['conversion_manifest_sha256']:raise ValueError('Wrong target quantization manifest')
    for name,entry in proof['files'].items():
        p=model/name
        if not p.is_file() or p.stat().st_size!=entry['bytes']:raise ValueError('Missing or changed model file: '+name)
        if (full or not name.endswith('.safetensors')) and sha(p)!=entry['sha256']:raise ValueError('Wrong model file: '+name)
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prefix',type=pathlib.Path);ap.add_argument('--model',type=pathlib.Path)
    ap.add_argument('--verify-only',action='store_true');ap.add_argument('--verify-weights',action='store_true')
    args=ap.parse_args();release=read(ROOT/'RELEASE.json')
    verify(ROOT,release['files'])
    for path,digest in release['foundation']['files'].items():
        if sha(path)!=digest:raise ValueError('Foundation differs: '+path)
    if args.verify_only:
        print(json.dumps({'verified':True,'files':len(release['files']),'foundation':release['foundation']['root']}));return
    if args.prefix is None or args.model is None:ap.error('--prefix and --model are required for installation')
    dest=args.prefix.absolute();model=args.model.resolve()
    if dest.exists():raise FileExistsError('Destination already exists; use a new release directory: '+str(dest))
    model_check(model,read(ROOT/'MODEL.json'),args.verify_weights)
    # Every source check precedes destination creation. A failed install is retained for diagnosis.
    dest.mkdir(parents=True)
    for name in ['payload','runtime-env.sh','serve.sh','audit_entry.py','VERIFY.py','MODEL.json','RELEASE.json','README.md','evidence','provenance']:
        source=ROOT/name
        if source.is_dir():shutil.copytree(source,dest/name,symlinks=True)
        elif source.exists():shutil.copy2(source,dest/name)
    mapping={key:str(dest/value) for key,value in release['relocations'].items()}
    texts={}
    for p in (dest/'payload').rglob('*'):
        if p.is_symlink():
            link=os.readlink(p);new=link
            for old,value in sorted(mapping.items(),key=lambda x:-len(x[0])):new=new.replace(old,value)
            if new!=link:p.unlink();p.symlink_to(new)
            continue
        if not p.is_file() or p.stat().st_size>4_000_000:continue
        if p.suffix not in ('.py','.json','.sh','.pth','.cfg','.txt','.h','.hip','.cpp') and p.name not in ('entry_points.txt',):continue
        try:text=p.read_text()
        except UnicodeDecodeError:continue
        texts[p]=text
    originals={p:sha(p) for p in texts}
    for p,text in texts.items():
        for old,value in sorted(mapping.items(),key=lambda x:-len(x[0])):text=text.replace(old,value)
        rewrite(p,text)
    # Source-integrity tables and embedded manifest digests form a dependency DAG.
    # Propagate only hashes of relocated text; kernel/weight bytes remain untouched.
    digest_map={}
    for iteration in range(20):
        current={originals[p]:sha(p) for p in texts if originals[p]!=sha(p)}
        changed=False
        for p in texts:
            text=p.read_text()
            def replace(match):
                key=match.group(0)
                # Resolve either the original digest or a prior intermediate digest.
                origin=digest_map.get(key,key)
                return current.get(origin,key)
            new=re.sub(r'\b[0-9a-f]{64}\b',replace,text)
            if new!=text:rewrite(p,new);changed=True
        for old,new in current.items():digest_map[new]=old
        if not changed:break
    else:raise RuntimeError('Integrity dependencies did not converge')
    mini=dest/'payload/model'
    deploy=mini/'candidate/minicpm/target-deployment.json'
    deploy.parent.mkdir(parents=True,exist_ok=True)
    proof=read(dest/'MODEL.json')
    save(deploy,dict(destination=str(model),conversion_manifest_sha256=proof['conversion_manifest_sha256'],verified_payload_files=len(proof['files'])))
    # This launcher keeps the retained ownership/locking behavior; only allow relocated target weights.
    launch=mini/'20261013-haloloom-gdn-remainder/launch.py'
    text=launch.read_text();old="assert str(model).startswith('/srv/llm/ciru-q8-20261012/'),model"
    if text.count(old)!=1:raise RuntimeError('Unexpected retained launcher target guard')
    rewrite(launch,text.replace(old,"assert model.is_dir(),model"))
    runenv=mini/'candidate/run-env.sh'
    rewrite(runenv,'#!/usr/bin/env bash\nset -euo pipefail\nroot=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)\nexec "$root/runtime-env.sh" "$@"\n')
    runenv.chmod(0o755)
    receipt=dict(release=release['name'],installed_root=str(dest),model=str(model),foundation=release['foundation'],
        model_weights_fully_verified=args.verify_weights,relocations=mapping,service_changed=False,profile_changed=False,
        transformations='Path relocation, dependent integrity digests, target-path guard and runtime environment launcher only.',
        original_source_pins=release['source_pins'],files=inventory(dest/'payload'))
    save(dest/'INSTALLATION.json',receipt)
    print(json.dumps({'installed':str(dest),'files':len(receipt['files']),'model':str(model),'service_changed':False}))
if __name__=='__main__':main()
