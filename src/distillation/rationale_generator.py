import os
import gc
import json
import torch
import logging
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

# few-shot CoT examples — specific to refugee/migrant misinformation domain
COT_EXAMPLES = [
    {
        "text": "Official statistics reveal that 80% of refugees return home within 5 years of arrival.",
        "reasoning": (
            "Step 1: This makes a specific statistical claim about refugee return rates.\n"
            "Step 2: UNHCR data shows the vast majority of refugees remain displaced for years or decades — "
            "this 80% figure contradicts official data.\n"
            "Step 3: No source is cited for the statistic.\n"
            "Step 4: The framing implies refugees are temporary and self-resolving, downplaying the crisis.\n"
            "Step 5: This is false information misrepresenting refugee displacement patterns."
        ),
        "label": "FALSE",
    },
    {
        "text": "Research from the IMF shows migrants contribute positively to host country GDP through taxes and entrepreneurship.",
        "reasoning": (
            "Step 1: Claims migrants have a positive economic impact.\n"
            "Step 2: This is consistent with IMF, OECD, and World Bank research on migration economics.\n"
            "Step 3: Cites a credible institution (IMF).\n"
            "Step 4: Language is neutral and evidence-based.\n"
            "Step 5: This is a factual statement supported by economic research."
        ),
        "label": "TRUE",
    },
    {
        "text": "Refugees are being secretly resettled in rural communities to change voting demographics.",
        "reasoning": (
            "Step 1: Claims a deliberate hidden agenda behind refugee resettlement.\n"
            "Step 2: Refugee resettlement decisions are made by UNHCR and national governments under public policy, not secretly.\n"
            "Step 3: The claim of demographic manipulation is a common conspiracy narrative with no factual basis.\n"
            "Step 4: Uses loaded language ('secretly') to imply sinister intent.\n"
            "Step 5: This is misinformation promoting a conspiracy theory."
        ),
        "label": "FALSE",
    },
]


def load_teacher_model(model_path, bits=4):
    print(f"Loading teacher model from {model_path}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    bnb = None
    if bits == 4:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    elif bits == 8:
        bnb = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()

    print("Teacher model loaded")
    return model, tokenizer


def _build_prompt(text, examples=None):
    if examples is None:
        examples = COT_EXAMPLES

    prompt = "Analyze this text about potential misinformation related to refugees or migrants.\n\n"
    prompt += "Examples:\n"

    for ex in examples:
        prompt += f'\nText: "{ex["text"]}"\nReasoning: {ex["reasoning"]}\nLabel: {ex["label"]}\n'

    prompt += f'\nNow analyze this text:\nText: "{text}"\nReasoning:'
    return prompt


def _parse_response(response, original_text):
    # strip the input from the response if echoed back
    if original_text in response:
        response = response.split(original_text, 1)[-1].strip()

    # try to find label
    label = None
    if "Label:" in response:
        parts = response.split("Label:", 1)
        rationale = parts[0].strip()
        label_raw = parts[1].strip().upper()
        label = "TRUE" if "TRUE" in label_raw else "FALSE"
    else:
        # fallback: look at last 3 sentences
        sentences = [s for s in response.split(".") if s.strip()]
        tail = ". ".join(sentences[-3:]).lower()
        rationale = response
        if "false" in tail or "misinformation" in tail or "mislead" in tail:
            label = "FALSE"
        else:
            label = "TRUE"

    return rationale.strip(), label


def _generate_one(model, tokenizer, text, max_new_tokens=300):
    prompt = _build_prompt(text)
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(model.device)

    with torch.no_grad():
        out = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
        )

    generated = tokenizer.decode(out[0], skip_special_tokens=True)
    return _parse_response(generated, text)


def generate_rationales(model, tokenizer, texts, labels, output_path=None, cache_path=None):
    """Generate step-by-step rationales from teacher model.

    Checks cache first to avoid regenerating — this is expensive.
    """
    # load from cache if available
    if cache_path and Path(cache_path).exists():
        print(f"Loading cached rationales from {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    print(f"Generating rationales for {len(texts)} texts...")
    results = []
    checkpoint_every = 10

    for i, (text, label) in enumerate(tqdm(zip(texts, labels), total=len(texts))):
        try:
            rationale, pred_label = _generate_one(model, tokenizer, text)
            results.append({
                "text": text,
                "rationale": rationale,
                "teacher_label": pred_label,
                "true_label": int(label),
            })
        except Exception as e:
            print(f"Failed on example {i}: {e}")
            results.append({
                "text": text,
                "rationale": "",
                "teacher_label": "ERROR",
                "true_label": int(label),
            })

        # save checkpoint periodically
        if output_path and (i + 1) % checkpoint_every == 0:
            with open(str(output_path) + f".checkpoint_{i+1}.json", "w") as f:
                json.dump(results, f, indent=2)

    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved rationales to {output_path}")

    # also save to cache path if provided
    if cache_path:
        with open(cache_path, "w") as f:
            json.dump(results, f, indent=2)

    labels_out = [r["teacher_label"] for r in results]
    print(f"Done. TRUE: {labels_out.count('TRUE')}, FALSE: {labels_out.count('FALSE')}, ERROR: {labels_out.count('ERROR')}")

    return results
