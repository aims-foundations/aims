"""
Curate the TUMLU benchmark into a response matrix and item_content.csv.

TUMLU: Turkic Multilingual Language Understanding benchmark
- 12 language variants (9 base languages, some with script variants)
- ~14 models x 2 prompting variants (CoT / no-CoT)
- Multiple subjects per language
"""

import json
import os
import re
import csv
import pandas as pd
from collections import defaultdict

BASE = "/lfs/skampere2/0/sttruong/aims/src/data/curated_benchmarks/tumlu_data/raw/TUMLU/data"
OUT = "/lfs/skampere2/0/sttruong/aims/src/data/curated_benchmarks/tumlu_data/processed"

# Model name normalization (full path -> short name)
MODEL_NAMES = {
    "Qwen/Qwen2.5-72B-Instruct": "qwen2.5-72b",
    "Qwen/Qwen2.5-7B-Instruct": "qwen2.5-7b",
    "claude-3-5-haiku-20241022": "claude-3.5-haiku",
    "claude-3-5-sonnet-20241022": "claude-3.5-sonnet",
    "deepseek-chat": "deepseek-chat",
    "gemini-1.5-flash": "gemini-1.5-flash",
    "gemini-1.5-pro": "gemini-1.5-pro",
    "google/gemma-2-27b-it": "gemma-2-27b",
    "google/gemma-2-9b-it": "gemma-2-9b",
    "gpt-4o-2024-11-20": "gpt-4o",
    "meta-llama/Llama-3.3-70B-Instruct": "llama-3.3-70b",
    "meta-llama/Meta-Llama-3.1-405B-Instruct": "llama-3.1-405b",
    "meta-llama/Meta-Llama-3.1-405B-Instruct-hyperbolic": "llama-3.1-405b-hyp",
    "meta-llama/Meta-Llama-3.1-405B-Instruct-deepinfra": "llama-3.1-405b-deepinfra",
    "meta-llama/Meta-Llama-3.1-70B-Instruct": "llama-3.1-70b",
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "llama-3.1-8b",
}


def extract_answer(output_text):
    """Extract the answer letter (A/B/C/D) from model output."""
    if not output_text or not isinstance(output_text, str):
        return None

    text = output_text.strip()

    # Pattern 1: Just the letter alone (with optional punctuation)
    if re.match(r'^[A-Da-d][\.\)\s]*$', text):
        return text[0].upper()

    # Pattern 2: "Cavab: X", "Answer: X", "Cevap: X", "Жауап: X", etc.
    m = re.search(
        r'(?:cavab|answer|cevap|жауап|jawap|جاۋاپ|جواب|javobi|җавап|жавап|жаваб|cawab|cevabı|düzgün cavab)\s*[:：]\s*\**\s*([A-Da-d])',
        text, re.IGNORECASE
    )
    if m:
        return m.group(1).upper()

    # Pattern 2b: "X variantıdır" / "X варианты" / "X varianti" (Turkic "is variant X")
    # Match last occurrence of this pattern
    matches_variant = re.findall(
        r'\b([A-Da-d])\s*(?:variantıdır|варианты|varianti|variantı)',
        text, re.IGNORECASE
    )
    if matches_variant:
        return matches_variant[-1].upper()

    # Pattern 2c: "cavab X variantıdır" without colon
    m = re.search(
        r'(?:cavab|cevap|жауап|jawap|düzgün cavab|doğru cevap|тоғры җавап)\s+\**([A-Da-d])\b',
        text, re.IGNORECASE
    )
    if m:
        return m.group(1).upper()

    # Pattern 3: **X)** or **X.**
    m = re.search(r'\*\*([A-Da-d])[\)\.]', text)
    if m:
        return m.group(1).upper()

    # Pattern 3b: **X** at the end or as standalone
    m = re.search(r'\*\*([A-Da-d])\*\*', text)
    if m:
        return m.group(1).upper()

    # Pattern 4: Standalone letter at start "A)" or "B." etc.
    m = re.match(r'\s*([A-Da-d])\s*[\)\.]', text)
    if m:
        return m.group(1).upper()

    # Pattern 5: Last occurrence of a letter option reference
    matches = re.findall(r'\b([A-Da-d])\)', text)
    if matches:
        return matches[-1].upper()

    # Pattern 6: First letter if very short response
    m = re.match(r'\s*([A-Da-d])\b', text)
    if m and len(text) < 20:
        return m.group(1).upper()

    return None


def normalize_subject(filename):
    """Normalize subject filename to a clean identifier."""
    name = filename.replace('.json', '').lower()
    name = re.sub(r'[^a-z0-9]', '_', name)
    name = re.sub(r'_+', '_', name).strip('_')
    return name


def normalize_language(lang):
    """Normalize language directory name."""
    return lang.lower().replace('-', '_')


def discover_models(variant_dir):
    """Discover all model paths under a variant directory.

    Some models are flat (e.g., gpt-4o-2024-11-20/Physics.json)
    while others are nested (e.g., Qwen/Qwen2.5-72B-Instruct/Physics.json).

    Returns list of (model_key, model_dir) tuples.
    """
    models = []
    if not os.path.exists(variant_dir):
        return models

    for entry in sorted(os.listdir(variant_dir)):
        entry_path = os.path.join(variant_dir, entry)
        if not os.path.isdir(entry_path):
            continue

        # Check if this directory contains JSON files directly
        contents = os.listdir(entry_path)
        has_json = any(f.endswith('.json') for f in contents)
        has_subdirs = any(os.path.isdir(os.path.join(entry_path, f)) for f in contents)

        if has_json:
            # Flat model directory
            models.append((entry, entry_path))

        if has_subdirs:
            # Nested: check subdirectories
            for sub in sorted(contents):
                sub_path = os.path.join(entry_path, sub)
                if os.path.isdir(sub_path):
                    sub_contents = os.listdir(sub_path)
                    if any(f.endswith('.json') for f in sub_contents):
                        model_key = f"{entry}/{sub}"
                        models.append((model_key, sub_path))

    return models


def main():
    languages = sorted(os.listdir(BASE))

    # Step 1: Register all items using a reference model per language/subject
    item_registry = {}  # item_id -> metadata
    item_order = []  # ordered list of item_ids

    for lang in languages:
        lang_norm = normalize_language(lang)
        ref_dir = os.path.join(BASE, lang, "outputs", "no_cot_instruct")
        if not os.path.exists(ref_dir):
            print(f"WARNING: No no_cot_instruct for {lang}")
            continue

        # Find a reference model that has the most subjects
        ref_models = discover_models(ref_dir)
        if not ref_models:
            print(f"WARNING: No models found for {lang}")
            continue

        # Use gpt-4o if available, else first model
        ref_key, ref_path = ref_models[0]
        for mk, mp in ref_models:
            if 'gpt-4o' in mk:
                ref_key, ref_path = mk, mp
                break

        for subj_file in sorted(os.listdir(ref_path)):
            if not subj_file.endswith('.json'):
                continue
            subj_norm = normalize_subject(subj_file)

            items = json.load(open(os.path.join(ref_path, subj_file)))
            for idx, item in enumerate(items):
                item_id = f"{lang_norm}_{subj_norm}_{idx}"

                question = item.get('question', '')
                choices = item.get('choices', [])
                choice_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
                choices_str = " | ".join(
                    f"{choice_labels[i]}) {c}" for i, c in enumerate(choices)
                )
                content = f"{question} [{choices_str}]"

                item_registry[item_id] = {
                    'question': question,
                    'choices': choices,
                    'answer': item.get('answer', ''),
                    'content': content,
                    'language': lang,
                    'subject': subj_norm,
                }
                item_order.append(item_id)

    print(f"Total items registered: {len(item_order)}")

    # Check if some models have subjects that our reference didn't have
    # Collect all possible subjects per language
    for lang in languages:
        lang_norm = normalize_language(lang)
        for variant in ["no_cot_instruct"]:
            vdir = os.path.join(BASE, lang, "outputs", variant)
            for mk, mp in discover_models(vdir):
                for subj_file in sorted(os.listdir(mp)):
                    if not subj_file.endswith('.json'):
                        continue
                    subj_norm = normalize_subject(subj_file)
                    test_id = f"{lang_norm}_{subj_norm}_0"
                    if test_id not in item_registry:
                        # This subject wasn't in our reference model - add items
                        items = json.load(open(os.path.join(mp, subj_file)))
                        for idx, item in enumerate(items):
                            item_id = f"{lang_norm}_{subj_norm}_{idx}"
                            if item_id in item_registry:
                                continue
                            question = item.get('question', '')
                            choices = item.get('choices', [])
                            choice_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
                            choices_str = " | ".join(
                                f"{choice_labels[i]}) {c}" for i, c in enumerate(choices)
                            )
                            content = f"{question} [{choices_str}]"
                            item_registry[item_id] = {
                                'question': question,
                                'choices': choices,
                                'answer': item.get('answer', ''),
                                'content': content,
                                'language': lang,
                                'subject': subj_norm,
                            }
                            item_order.append(item_id)

    print(f"Total items after filling gaps: {len(item_order)}")

    # Step 2: Build response matrix
    response_data = {}  # model_label -> {item_id: 0/1}

    for variant in ["no_cot_instruct", "cot_instruct"]:
        variant_suffix = "" if variant == "no_cot_instruct" else "_cot"

        for lang in languages:
            lang_norm = normalize_language(lang)
            variant_dir = os.path.join(BASE, lang, "outputs", variant)

            for model_key, model_path in discover_models(variant_dir):
                model_name = MODEL_NAMES.get(model_key, model_key) + variant_suffix

                if model_name not in response_data:
                    response_data[model_name] = {}

                for subj_file in sorted(os.listdir(model_path)):
                    if not subj_file.endswith('.json'):
                        continue
                    subj_norm = normalize_subject(subj_file)

                    items = json.load(open(os.path.join(model_path, subj_file)))
                    for idx, item in enumerate(items):
                        item_id = f"{lang_norm}_{subj_norm}_{idx}"

                        if item_id not in item_registry:
                            continue

                        gold = item.get('answer', '').strip().upper()
                        predicted = extract_answer(item.get('output', ''))

                        correct = 1 if (predicted is not None and predicted == gold) else 0
                        response_data[model_name][item_id] = correct

    # Step 3: Build DataFrame
    models = sorted(response_data.keys())

    matrix = []
    for model in models:
        row = [model]
        for item_id in item_order:
            val = response_data[model].get(item_id, '')
            row.append(val)
        matrix.append(row)

    columns = ['model'] + item_order
    df = pd.DataFrame(matrix, columns=columns)

    # Save response matrix
    df.to_csv(os.path.join(OUT, "response_matrix.csv"), index=False)
    print(f"\nResponse matrix saved: {df.shape[0]} models x {df.shape[1]-1} items")

    # Step 4: Build item_content.csv
    with open(os.path.join(OUT, "item_content.csv"), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['item_id', 'content'])
        for item_id in item_order:
            writer.writerow([item_id, item_registry[item_id]['content']])
    print(f"Item content saved: {len(item_order)} items")

    # Step 5: Summary statistics
    print("\n" + "="*60)
    print("TUMLU BENCHMARK CURATION SUMMARY")
    print("="*60)

    print(f"\nResponse matrix shape: {df.shape[0]} models x {df.shape[1]-1} items")

    # Count non-empty entries per model
    print(f"\nModels ({len(models)}):")
    for m in models:
        n_answered = sum(1 for item_id in item_order if response_data[m].get(item_id, '') != '')
        print(f"  - {m}: {n_answered} items answered")

    print(f"\nLanguages ({len(languages)}):")
    lang_items = defaultdict(int)
    for item_id in item_order:
        lang = item_registry[item_id]['language']
        lang_items[lang] += 1
    for lang in sorted(lang_items):
        print(f"  - {lang}: {lang_items[lang]} items")

    print(f"\nSubjects per language:")
    for lang in sorted(lang_items):
        subjects = set()
        for item_id in item_order:
            if item_registry[item_id]['language'] == lang:
                subjects.add(item_registry[item_id]['subject'])
        print(f"  - {lang}: {sorted(subjects)}")

    # Overall accuracy
    print(f"\nOverall accuracy by model:")
    for model in models:
        vals = [response_data[model][item_id] for item_id in item_order
                if item_id in response_data[model]]
        if vals:
            acc = sum(vals) / len(vals)
            print(f"  {model}: {acc:.3f} ({len(vals)} items)")

    # Answer extraction rate
    total_entries = 0
    extracted = 0
    for variant in ["no_cot_instruct", "cot_instruct"]:
        for lang in languages:
            variant_dir = os.path.join(BASE, lang, "outputs", variant)
            for model_key, model_path in discover_models(variant_dir):
                for subj_file in sorted(os.listdir(model_path)):
                    if not subj_file.endswith('.json'):
                        continue
                    items = json.load(open(os.path.join(model_path, subj_file)))
                    for item in items:
                        total_entries += 1
                        if extract_answer(item.get('output', '')) is not None:
                            extracted += 1

    print(f"\nAnswer extraction rate: {extracted}/{total_entries} = {extracted/total_entries:.1%}")


if __name__ == "__main__":
    main()
