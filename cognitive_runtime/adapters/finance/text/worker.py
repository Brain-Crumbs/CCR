"""Private worker entry point. Optional imports occur only after hard limits."""
import json
import sys
import time


def main():
    import resource
    started = time.monotonic()
    # Linux/POSIX only, fail rather than silently run without limits.
    if sys.platform != 'linux':
        raise RuntimeError('bounded local worker requires Linux')
    resource.setrlimit(resource.RLIMIT_AS, (4*1024**3, 4*1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (50, 50))
    attempts = []
    def offline(event, args):
        if event in ('socket.connect', 'socket.getaddrinfo', 'socket.sendto', 'socket.bind'):
            attempts.append(event)
            raise RuntimeError('local worker network denied')
    sys.addaudithook(offline)
    request = json.loads(sys.stdin.read(65537))
    from .local import verify_staged, SETTINGS
    from .extractor import prompt_for, validate_output
    root = verify_staged(request['pin'], request['staged'])
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
            output = torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().float().tolist()
        else:
            text = request['texts'][0]
            prompt = tokenizer.apply_chat_template(prompt_for(text), tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors='pt', truncation=False)
            if inputs['input_ids'].shape[1] > settings['max_tokens']:
                raise ValueError('prompt exceeds token bound; narrow text upstream')
            generated = model.generate(**inputs, max_new_tokens=384, do_sample=False,
                                       use_cache=True, pad_token_id=tokenizer.eos_token_id)
            output = tokenizer.decode(generated[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            validate_output(output, text)
    if attempts:
        raise RuntimeError('network attempt even if caught by library')
    print(json.dumps({'result': output, 'network_attempts': 0,
        'metrics': {'runtime_seconds': time.monotonic()-started,
                    'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    'address_space_limit_bytes': 4*1024**3, 'cpu_limit_seconds': 50, 'threads': 2}}))


if __name__ == '__main__':
    main()
