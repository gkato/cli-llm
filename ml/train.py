from ml.adapters import save_metadata
from ml.config import DATA, MODELS
from ml.datasets import split, validate


def _format_alpaca(row: dict, eos: str) -> dict:
    instr = row["instruction"]
    inp = row.get("input", "")
    out = row["output"]
    text = f"### Instruction:\n{instr}\n\n"
    if inp:
        text += f"### Input:\n{inp}\n\n"
    text += f"### Response:\n{out}{eos}"
    return {"text": text}


def train(
    model_key: str,
    dataset_key: str,
    adapter_name: str,
    epochs: int = 3,
    lr: float = 2e-4,
    rank: int = 16,
    max_seq_length: int = 2048,
    batch_size: int = 2,
    grad_accum: int = 4,
):
    import torch
    from datasets import Dataset
    from transformers import TrainingArguments
    from trl import SFTTrainer
    from unsloth import FastLanguageModel

    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}")

    info = validate(dataset_key)
    print(f"Dataset {dataset_key}: {info['count']} rows ({info['format']})")
    train_rows, val_rows, _ = split(dataset_key)
    print(f"Splits — train={len(train_rows)} val={len(val_rows)}")

    repo = MODELS[model_key]["unsloth_repo"]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=repo,
        max_seq_length=max_seq_length,
        dtype=torch.float16,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=rank,
        lora_alpha=rank * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    eos = tokenizer.eos_token
    train_ds = Dataset.from_list([_format_alpaca(r, eos) for r in train_rows])
    val_ds = Dataset.from_list([_format_alpaca(r, eos) for r in val_rows])

    output_dir = DATA / "adapters" / adapter_name
    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        warmup_steps=20,
        learning_rate=lr,
        logging_steps=10,
        save_steps=200,
        eval_strategy="steps",
        eval_steps=100,
        save_total_limit=2,
        fp16=True,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=args,
        max_seq_length=max_seq_length,
        dataset_text_field="text",
    )
    result = trainer.train()

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    save_metadata(
        adapter_name,
        base_model=model_key,
        unsloth_repo=repo,
        dataset=dataset_key,
        epochs=epochs,
        lr=lr,
        rank=rank,
        final_loss=float(result.training_loss),
    )
    print(f"✓ Adapter saved to {output_dir}")
    return output_dir


def infer_with_adapter(adapter_name: str, prompt: str, max_new_tokens: int = 256) -> str:
    """Run inference against a trained adapter (transformers + peft, not Ollama)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ml.adapters import path_for

    adapter_path = path_for(adapter_name)
    meta_path = adapter_path / "metadata.json"
    if not meta_path.exists():
        raise RuntimeError(f"Adapter {adapter_name} missing metadata.json")
    import json
    meta = json.loads(meta_path.read_text())
    base_repo = meta["unsloth_repo"]

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
    base = AutoModelForCausalLM.from_pretrained(
        base_repo,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_4bit=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()

    text = f"### Instruction:\n{prompt}\n\n### Response:\n"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0], skip_special_tokens=True)
