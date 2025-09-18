# molmo_test_local.py
import os, torch, io, requests
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
from pathlib import Path

LOCAL_DIR = r"C:\models\MolmoE-1B-0924"

# ensure we aren't blocking image processing / downloads
os.environ.pop("TRANSFORMERS_NO_TORCHVISION", None)

def to_device_batch(d, device):
    out = {}
    for k, v in d.items():
        if v is None: continue
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device).unsqueeze(0)
        else:
            out[k] = v
    return out

def main():
    assert Path(LOCAL_DIR).exists(), "Missing model dir"
    assert any(p.suffix==".safetensors" for p in Path(LOCAL_DIR).glob("*.safetensors")), \
        "No .safetensors found in LOCAL_DIR"

    processor = AutoProcessor.from_pretrained(
        LOCAL_DIR, trust_remote_code=True, local_files_only=True, use_fast=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_DIR,
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.float16,
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )

    # test image
    r = requests.get("https://picsum.photos/id/237/536/354", timeout=20)
    img = Image.open(io.BytesIO(r.content)).convert("RGB")

    inputs = processor.process(images=[img], text="Describe this image.")
    inputs = to_device_batch(inputs, next(model.parameters()).device)

    out = model.generate_from_batch(
        inputs,
        GenerationConfig(max_new_tokens=80, stop_strings="<|endoftext|>",
                         pad_token_id=processor.tokenizer.eos_token_id),
        tokenizer=processor.tokenizer,
    )
    prompt_len = inputs["input_ids"].size(1)
    print(processor.tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True).strip())

if __name__ == "__main__":
    main()
