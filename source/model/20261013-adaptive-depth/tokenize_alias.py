"""Normalize tokenization using generation's exact installed validator."""
import copy, json, os, pathlib, sys
from pydantic import model_validator

def install():
    assert 'vllm.entrypoints.serve.tokenize.api_router' not in sys.modules
    assert 'vllm.entrypoints.serve.tokenize.serving' not in sys.modules
    from vllm.entrypoints.serve.tokenize import protocol
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    Parent=protocol.TokenizeChatRequest
    class TokenizeChatRequest(Parent):
        @model_validator(mode='before')
        @classmethod
        def normalize_generation_aliases(cls,data):
            return ChatCompletionRequest._normalize_messages_before(copy.deepcopy(data))
    protocol.TokenizeChatRequest=TokenizeChatRequest
    protocol.TokenizeRequest=protocol.TokenizeCompletionRequest | TokenizeChatRequest
    root=pathlib.Path(os.environ['CIRU_SERVING_RUN'])
    (root/'tokenize-binding.json').write_text(json.dumps(dict(patch=__file__,generation_validator='ChatCompletionRequest._normalize_messages_before',generation_changed=False,protocol=protocol.__file__,api_router_loaded_before_patch=False))+'\n')
