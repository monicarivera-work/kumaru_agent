# Training and fine-tuning for Kumaru

What it actually takes to make a model that is *yours*, and how to serve it
with Kumaru afterwards. Read this before you spend money on GPUs.

---

## 0. The decision that saves you the most time

Most people who think they need training need one of the cheaper options
first. In order:

| Option | Cost | Time | Use when |
|---|---|---|---|
| **Change the system prompt** | free | minutes | You want a different tone, format, or persona. |
| **Retrieval (RAG)** | small | hours | The model needs *facts* it does not have. Facts change. |
| **LoRA fine-tune** | $1–50 | hours | You need a consistent *behaviour*, format, or domain style. |
| **Full fine-tune** | $100s–1000s | days | LoRA measurably is not enough. Rare. |
| **Pre-training from scratch** | $100k+ | weeks–months | Essentially never, outside research labs. |

Rules of thumb that hold up in practice:

* **New knowledge → retrieval.** Fine-tuning is a bad database: it is expensive
  to update and it hallucinates confidently around the edges.
* **New behaviour → fine-tuning.** "Always answer as a JSON object with these
  keys" is a behaviour, and it is exactly what LoRA is good at.
* You cannot fine-tune a small model into a large one. A 3B model with a
  perfect fine-tune still reasons like a 3B model.

## 1. What a LoRA fine-tune actually costs

LoRA (Low-Rank Adaptation) freezes the base model and trains small adapter
matrices — typically well under 1% of the parameters. That is what makes it
affordable.

| Base model | Method | Min VRAM | 1k examples, 3 epochs | Adapter size |
|---|---|---|---|---|
| 1–3B | LoRA bf16 | 12 GB | ~15–30 min | 10–50 MB |
| 7–8B | QLoRA 4-bit | 12 GB | ~1–2 h | 30–80 MB |
| 7–8B | LoRA bf16 | 24 GB | ~1 h | 30–80 MB |
| 13–14B | QLoRA 4-bit | 24 GB | ~2–4 h | 60–150 MB |
| 70B | QLoRA 4-bit | 80 GB (A100/H100) | ~1–2 days | 150–400 MB |

Rented GPUs are roughly $0.40–$2.50/hour, so a 7B QLoRA run is usually a few
dollars. Budget more for the *iterations*: nobody's first dataset is right.

## 2. The dataset is the whole job

Expect to spend 80% of your effort here. It decides the outcome far more than
any hyperparameter.

* **Volume:** 500–1,000 good examples beat 50,000 scraped ones. Start at ~1,000.
* **Consistency:** every example must demonstrate the behaviour you want. One
  contradictory batch teaches the model to be inconsistent.
* **Diversity:** cover the real distribution of requests, including the awkward
  ones and the ones that should be refused.
* **Hold-out:** reserve 5–10% that training never sees. Without it you cannot
  tell improvement from memorisation.

Format — JSONL, one conversation per line, matching Kumaru's message shape:

```json
{"messages": [{"role": "system", "content": "You are Kumaru."}, {"role": "user", "content": "Summarise this incident report: ..."}, {"role": "assistant", "content": "..."}]}
```

Keep the `system` message identical to the one you will run in production
(`agent.system_prompt`). Training and serving prompts that differ is the most
common cause of "it was great in training and useless in the app".

## 3. Running the fine-tune

Kumaru does not reimplement a trainer — writing one badly is how people waste a
month. Use a maintained tool, then load the result:

| Tool | Good for |
|---|---|
| [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) | YAML-driven LoRA/QLoRA; sensible defaults |
| [Unsloth](https://github.com/unslothai/unsloth) | 2× faster, lower VRAM, single GPU |
| [TRL](https://github.com/huggingface/trl) (`SFTTrainer`) | Most control; plain Python |
| [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) | Many models, a UI |

Hyperparameters that matter, with defaults that usually work:

| Parameter | Start at | Notes |
|---|---|---|
| LoRA rank `r` | 16 | 8 for style, 32–64 for harder domain shifts |
| `lora_alpha` | 32 | conventionally `2 × r` |
| learning rate | 2e-4 | LoRA tolerates far higher LRs than full fine-tuning |
| epochs | 3 | 1–2 for >10k examples; watch for memorisation |
| batch size | as large as fits | use gradient accumulation to simulate 16–32 |
| max sequence length | 1024–2048 | VRAM grows with the square of this |

Watch the **validation** loss, not the training loss. Training loss falling
while validation loss rises means you are memorising; stop and take the earlier
checkpoint.

## 4. Serving your model with Kumaru

### A LoRA adapter merged into the base model

Most trainers can merge the adapter back into full weights. Then:

```yaml
# configs/my-model.yaml
backend:
  name: transformers
  model: ./models/my-finetune      # a directory of merged weights
  device: auto
```

```bash
kumaru serve -c configs/my-model.yaml
```

### The same model, quantised for CPU

Convert the merged model to GGUF with `llama.cpp`'s `convert_hf_to_gguf.py`,
quantise to Q4_K_M, then:

```yaml
backend:
  name: llamacpp
  model: models/my-finetune-q4_k_m.gguf
```

### Served by something else

Load it in Ollama or vLLM and point Kumaru at the URL — `configs/openai-compat.yaml`.

## 5. Did it work?

Decide this *before* you train, or you will grade your own homework.

1. Write 30–50 held-out prompts that represent real use.
2. Record the base model's answers **before** training. This is your baseline;
   without it every result looks impressive.
3. After training, generate the same answers and compare blind — you or a
   stronger model as judge.
4. Check for regressions: run a handful of general-knowledge and instruction-
   following prompts. Catastrophic forgetting is real, and a model that nails
   your domain while failing at basic requests is usually a net loss.

`kumaru ask` makes step 3 scriptable:

```bash
while read -r q; do
  echo "## $q"
  kumaru ask --no-stream -c configs/my-model.yaml "$q"
done < eval_prompts.txt > results.md
```

## 6. Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Output is unchanged | LoRA not actually loaded, or LR too low | verify the adapter path; try LR 2e-4 |
| Model repeats the training data verbatim | over-fitting | fewer epochs, more data, lower LR |
| Good on the new task, worse at everything | catastrophic forgetting | lower LR, fewer epochs, mix in general examples |
| Degenerate/garbled output | prompt template mismatch | use the same chat template in training and serving |
| CUDA out of memory | sequence length or batch too large | shorter sequences, 4-bit, gradient accumulation |
| Trains fine, useless in the app | training prompt ≠ `agent.system_prompt` | make them identical |

## 7. Honest expectations

A LoRA on a 7B model gives you a model that follows *your* format and speaks
*your* domain's language reliably. It does not give you a frontier model. The
fastest route to better answers, in order: a better base model, then retrieval,
then a fine-tune — and only then a bigger fine-tune.

See [`SCALABILITY.md`](SCALABILITY.md) for how to serve whatever you end up
with.
