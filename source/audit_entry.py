"""Qualification entrypoint: prohibit research-code fallback and record imported origins."""
import atexit,json,os,pathlib,runpy,sys
root=pathlib.Path(os.environ['CIRU_PACKAGE_ROOT'])
receipt=json.loads((root/'INSTALLATION.json').read_text())
blocked=tuple(receipt['relocations'])
def audit(event,args):
    if event in ('open','ctypes.dlopen') and args and isinstance(args[0],(str,bytes)):
        path=os.fsdecode(args[0])
        if path.startswith(blocked):raise RuntimeError('Research dependency blocked: '+path)
sys.addaudithook(audit)
def save():
    modules={k:str(getattr(m,'__file__','')) for k,m in list(sys.modules.items()) if getattr(m,'__file__',None)}
    (root/'state/import-origins.json').write_text(json.dumps(modules,indent=2)+'\n')
atexit.register(save)
sys.argv=sys.argv[1:]
runpy.run_path(sys.argv[0],run_name='__main__')
