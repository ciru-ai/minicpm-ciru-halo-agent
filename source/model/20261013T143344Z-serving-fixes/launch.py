"""Own one isolated adoption server under the existing Ciru GPU locks."""
import argparse,fcntl,hashlib,json,os,pathlib,re,shutil,signal,subprocess,sys,time
ap=argparse.ArgumentParser();ap.add_argument('model',choices=['apodex','minicpm']);ap.add_argument('--port',type=int,default=8095);ap.add_argument('--host',default='127.0.0.1');ap.add_argument('--teacher',action='store_true');ap.add_argument('--drafts',type=int,choices=[0,1,2,3,4,8,12,15],default=0);ap.add_argument('--variant');ap.add_argument('--managed-service',action='store_true');ap.add_argument('--inspect-only',action='store_true');ap.add_argument('--attention',choices=['stock','audit','retained'],default='stock');ap.add_argument('--prefill',choices=['stock','audit','retained'],default='stock');a=ap.parse_args()
assert a.model=='minicpm' and not a.teacher
a.adaptive=os.environ.get('CIRU_DSPARK_ADAPTIVE','0')=='1'
if a.variant:assert re.fullmatch(r'[a-zA-Z0-9_-]+',a.variant)
root=pathlib.Path(__file__).resolve().parent.parent/'candidate'
kv_gib=int(os.environ.get('CIRU_DSPARK_KV_GIB','24'));assert kv_gib in (16,24,32)
adapter_manifest=json.loads((root/'adapter-manifest.json').read_text())
for name,digest in adapter_manifest.items():
 assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,'Adapter source mismatch: '+name
locks=[]
for name in (() if a.inspect_only else ('/tmp/ciru-llm-gpu.lock','/tmp/ciru-qtip-lab.lock')):
 f=open(name,'a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
unit=subprocess.run(['systemctl','--user','is-active','qwen-main.service'],capture_output=True,text=True)
if not a.inspect_only:
 if a.managed_service:
  assert 'qwen-main.service' in pathlib.Path('/proc/self/cgroup').read_text(),'Managed mode requires the main service cgroup'
 elif unit.stdout.strip()!='inactive':raise RuntimeError('Main service must already be inactive; adoption does not change it')
run=root/a.model/(a.variant or ('teacher-runtime' if a.teacher else 'runtime'))
if not a.inspect_only:run.mkdir(exist_ok=True)
adapter=run/'adapter'
if not a.inspect_only:
 adapter.mkdir(exist_ok=True)
 for name in adapter_manifest:
  dst=adapter/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/name,dst)
 (run/'as-run-adapter-manifest.json').write_text(json.dumps(adapter_manifest,indent=2)+'\n')
if not a.inspect_only and (run/'server.log').exists():
 archive=run/'prior-launches'/str(time.time_ns());archive.mkdir(parents=True)
 for name in ('server.log','owner.json','command.json','exit.json','native-calls.jsonl','stop-receipt.json'):
  p=run/name
  if p.exists():p.rename(archive/name)
model=root/a.model/('source-bf16' if a.teacher else 'target-q8')
deployment=root/a.model/'target-deployment.json'
if not a.teacher and deployment.exists():
 info=json.loads(deployment.read_text());model=pathlib.Path(info['destination'])
 assert str(model).startswith('/srv/llm/ciru-q8-20261012/'),model
 assert hashlib.sha256((model/'conversion-manifest.json').read_bytes()).hexdigest()==info['conversion_manifest_sha256']
 assert info['verified_payload_files']>0
if not a.teacher:assert (model/'conversion-manifest.json').exists()
cmd=[str(root/'run-env.sh'),str(pathlib.Path(__file__).with_name('serve.py')),'--model',str(model),'--host',a.host,'--port',str(a.port),
 '--served-model-name',a.model+'-q8' if not a.teacher else a.model+'-bf16','--dtype','bfloat16','--safetensors-load-strategy','eager',
 '--max-model-len','262144','--max-num-seqs','8','--max-num-batched-tokens','8192',
 '--kv-cache-memory-bytes',str(kv_gib<<30),'--mamba-ssm-cache-dtype','float32','--no-async-scheduling',
 '--enable-prefix-caching','--mamba-cache-mode','align','--skip-mm-profiling',
 '--compilation-config',json.dumps({'mode':3,'cudagraph_mode':'FULL_DECODE_ONLY','cudagraph_capture_sizes':sorted({(a.drafts+1)*n for n in [1,2,4,8]} | ({a.drafts*n for n in [1,2,4,8]} if a.drafts else set()))}),
 '--mm-encoder-attn-backend','TRITON_ATTN','--attention-backend','ROCM_ATTN',
 '--enable-auto-tool-choice','--tool-call-parser','qwen3_coder','--reasoning-parser','qwen3',
 '--generation-config','auto','--seed','15035','--max-logprobs','20']
if a.drafts:cmd+=['--speculative-config',json.dumps({'method':'dspark','model':str(root.parent/'draft-bf16'),'num_speculative_tokens':a.drafts,'disable_eagle_block_drop':os.environ.get('CIRU_DSPARK_KEEP_PREFIX_TAIL','0')=='1','enable_adaptive_verification':a.adaptive,'draft_sample_method':os.environ.get('CIRU_DSPARK_SAMPLE','greedy')})]
if not a.teacher:cmd+=['--quantization','ciru_q8','--worker-cls','serving_worker.Worker',
 '--additional-config',json.dumps({'ciru_q8':{'run':str(run),'attention':a.attention,'prefill':a.prefill,'draft_attention':os.environ.get('CIRU_DSPARK_ATTENTION','retained')}})]
if os.environ.get('CIRU_BF16_PROFILE')=='1':
 cmd=[shutil.which('bash'),str(root.parent/'profile-env.sh'),'/opt/ciru/glm53-iu4/venv/bin/python','/opt/ciru/glm53-iu4/venv/bin/rocprofv3','--kernel-trace','--hip-trace','--marker-trace','--stats','--selected-regions','--output-format','csv','json','--output-directory',str(run/'rocprof'),'--output-file','bf16prefill','--']+cmd
if a.inspect_only:
 print(json.dumps(dict(command=cmd,run=str(run),context=262144,kv_gib=kv_gib,slots=8,native_mtp=a.drafts,main_changed=False,inspection_only=True),indent=2));sys.exit(0)
(run/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
(run/'attention-child-launch-sources.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [pathlib.Path(__file__),root.parent/'attention_candidate.py',root.parent/'attention_shared_query.py']},indent=2)+'\n')
env=os.environ.copy();env['PYTHONPATH']=str(pathlib.Path(__file__).parent)+':'+str(root.parent)+':'+env.get('PYTHONPATH','');env['VLLM_CACHE_ROOT']=str(root/a.model/('cache-attention-child-'+os.environ['CIRU_ATTENTION_CHILD']));env['TRITON_CACHE_DIR']=str(root/a.model/'triton-cache');env['CIRU_Q8_ADAPTER']=str(adapter)
for child in ('gdn_worker.py','gdn_spatial_worker.py','gdn_common.py','bf16_prefill_worker.py','bf16_prefill_kernel.py','bf16_prefill_native.py'):
 shutil.copyfile(root.parent/child,run/child)
native=run/'bf16-prefill-native-v1';native.mkdir(exist_ok=True)
for filename in ('prefill.hip','prefill.so','prefill.s','build-receipt.json','source.json','LICENSE.upstream','ATTRIBUTION.json'):
 shutil.copyfile(root.parent/'bf16-prefill-native-v1'/filename,native/filename)
with (run/'server.log').open('w') as log:
 p=subprocess.Popen(cmd,stdout=log,stderr=log,env=env,start_new_session=True)
 (run/'owner.json').write_text(json.dumps({'owner_pid':os.getpid(),'pid':p.pid,'command':cmd,'main_changed':False})+'\n')
 def stop(signum,frame):
  if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 try:code=p.wait()
 finally:
  if p.poll() is None:os.killpg(p.pid,signal.SIGTERM);p.wait()
 (run/'exit.json').write_text(json.dumps({'exit_code':code,'main_restored':False})+'\n')
 sys.exit(code)
