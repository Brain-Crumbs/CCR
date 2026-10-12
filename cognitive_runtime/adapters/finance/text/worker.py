"""Private worker entry point. Optional imports occur only after hard limits."""
import json
import sys
import time

MAX_REQUEST_BYTES = 256*1024


def read_request(stream):
    """Bytes framing accounts for UTF-8 and JSON escaping, not just code points."""
    raw = stream.read(MAX_REQUEST_BYTES + 1)
    if len(raw) > MAX_REQUEST_BYTES:
        raise ValueError('local model request exceeds 256 KiB')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate request key')
            result[key] = value
        return result
    request = json.loads(raw, object_pairs_hook=unique)
    from .local import validate_pin, MODELS
    from .features import MAX_CHARS
    if type(request) is not dict or set(request) != {'model', 'pin', 'staged', 'texts'}:
        raise ValueError('invalid local worker request fields')
    model, texts = request['model'], request['texts']
    if type(model) is not str or model not in MODELS:
        raise ValueError('unsupported worker model')
    if type(request['staged']) is not str or not request['staged'].startswith('/') or len(request['staged']) > 4096:
        raise ValueError('expected bounded absolute staged path')
    limit = 8 if model == 'minilm' else 1
    if type(texts) is not list or not 1 <= len(texts) <= limit or any(type(t) is not str or len(t) > MAX_CHARS for t in texts):
        raise ValueError('worker text/batch bound exceeded')
    validate_pin(request['pin'], model)
    return request


def enforce_limits():
    import resource
    if sys.platform != 'linux':
        raise RuntimeError('bounded local worker requires Linux')
    resource.setrlimit(resource.RLIMIT_AS, (4*1024**3, 4*1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (50, 50))
    attempts = []
    def offline(event, args):
        if event in ('socket.connect', 'socket.getaddrinfo', 'socket.gethostbyname',
                     'socket.gethostbyaddr', 'socket.sendto', 'socket.sendmsg', 'socket.bind'):
            attempts.append(event)
            raise RuntimeError('local worker network denied')
    sys.addaudithook(offline)
    return attempts


def infer(request, root):
    """Only optional-model computation; separated to test the real worker shell."""
    from .local import SETTINGS
    from .extractor import prompt_for, validate_output
    import torch
    from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
    torch.set_num_threads(2)
    kwargs = {'local_files_only': True, 'trust_remote_code': False}
    tokenizer = AutoTokenizer.from_pretrained(str(root), **kwargs)
    model_name = request['model']
    settings = SETTINGS[model_name]
    loader = AutoModel if model_name == 'minilm' else AutoModelForCausalLM
    model = loader.from_pretrained(str(root), use_safetensors=True, torch_dtype=torch.float32, **kwargs).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with torch.inference_mode():
        if model_name == 'minilm':
            inputs = tokenizer(request['texts'], padding=True, truncation=True, max_length=256, return_tensors='pt')
            hidden = model(**inputs).last_hidden_state
            mask = inputs['attention_mask'].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden*mask).sum(1)/mask.sum(1).clamp(min=1)
            return torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().float().tolist()
        text = request['texts'][0]
        prompt = tokenizer.apply_chat_template(prompt_for(text), tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors='pt', truncation=False)
        if inputs['input_ids'].shape[1] > settings['max_tokens']:
            raise ValueError('prompt exceeds token bound; narrow text upstream')
        generated = model.generate(**inputs, max_new_tokens=384, do_sample=False,
                                   use_cache=True, pad_token_id=tokenizer.eos_token_id)
        output = tokenizer.decode(generated[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        validate_output(output, text)
        return output


def main():
    import resource
    started = time.monotonic()
    attempts = enforce_limits()
    request = read_request(sys.stdin.buffer)
    from .local import verify_staged
    root = verify_staged(request['pin'], request['staged'])
    output = infer(request, root)
    if attempts:
        raise RuntimeError('network attempt even if caught by library')
    print(json.dumps({'result': output, 'network_attempts': 0,
        'metrics': {'runtime_seconds': time.monotonic()-started,
                    'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    'address_space_limit_bytes': resource.getrlimit(resource.RLIMIT_AS)[0],
                    'cpu_limit_seconds': resource.getrlimit(resource.RLIMIT_CPU)[0], 'threads': 2}}))


if __name__ == '__main__':
    main()
