"""One candidate: distribute the retained stage1 over eight wave32 warps."""
import hashlib
import inspect
from pathlib import Path

import attention_shared_query as retained

assert hashlib.sha256(Path(retained.__file__).read_bytes()).hexdigest() == '3e19c89ba36809c82c820c4dbebbb9efc43286fbbac9252632dc4ecca437c23c'
source = inspect.getsource(retained.forward)
needle = 'num_warps=4,num_stages=1'
assert source.count(needle) == 1
namespace = dict(retained.forward.__globals__)
exec(compile(source.replace(needle, 'num_warps=8,num_stages=1'), __file__ + '.forward', 'exec'), namespace)
forward = namespace['forward']
