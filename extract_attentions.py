"""
Run a causal LM on poem/prose prompt pairs and extract its attention weights.

For each prompt the model greedily decodes a response; at every decoding step the
attention weights of all layers and heads are recorded.

Usage (from the repository root):

    # smoke test on the example pair: nothing is written
    python extract_attentions.py --model_path Qwen/Qwen3-14B \
        --raw_input data/input/example.jsonl

    # full extraction from an mlcommons CSV directory
    python extract_attentions.py --model_path Qwen/Qwen3-14B \
        --raw_input <mlcommons_csv_dir> --split train --save_responses --save_attentions

Input (--raw_input), one of:

    JSONL file   one poem/prose pair per line:
                 {"release_prompt_id": ..., "poetry_text": ..., "prose_text": ...}
    CSV dir      mlcommons/AILuminate run with baseline_fail.csv, baseline_success.csv
                 (prose) and poetic_fail.csv, poetic_success.csv (poetry); columns used:
                 prompt_id ("<variant>::<sample_id>"), prompt_text, safety_status, hazard

Output (extraction_results/<model>[_<suffix>]/<split>/<sample_id>/<poetry|prose>/):

    response.json     with --save_responses: prompt tokens, model response, timing
                      (+ prompt_id, safety label and hazard for CSV input)
    attentions.json   with --save_attentions: per decoding step, the generated token and
                      the attention weights of every layer, [layer][batch][head][query][key]

A prompt is skipped when every requested file already exists, and an existing
response.json is never overwritten (later pipeline steps add labels to it).
"""

import os 
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import argparse
from datetime import datetime, timezone
from collections import defaultdict
import time
import torch
import csv

# Greedy decoding length. Changing it changes the extracted attentions.
MAX_NEW_TOKENS = 50

class AttentionExtractor:
    """
    Loads a model and extracts its attention weights while it answers each prompt.
    """
    def __init__(self,
                model_name='',
                model_name_suffix=None,
                save_responses = False,
                save_attentions = False,
                print_ongoing_status = True):
        """        
        :param model_name: HF model ID or local path of the model to load.
        :param model_name_suffix: Suffix appended to the output directory name
            (extraction_results/<model>_<suffix>/...). Default is None.
        :param save_responses: Whether to write response.json for each prompt. Default is False.
        :param save_attentions: Whether to write attentions.json for each prompt. Default is False.
            With both False nothing is written (e.g. for smoke tests).
        :param print_ongoing_status: Whether or not to print the ongoing status. Default is True.
        """
        self.save_responses = save_responses
        self.save_attentions = save_attentions
        self.print_ongoing_status = print_ongoing_status
        if("/" in model_name):
            self.model_version = model_name.split("/")[-1]
        else: self.model_version = model_name
        if(model_name_suffix is not None): self.model_version += "_" + model_name_suffix


        self.enc = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        print("loading from %s" % model_name)
        config = AutoConfig.from_pretrained(model_name)
        
        # Handle different config structures (multimodal models like Qwen3.5 have text_config)
        if hasattr(config, 'text_config'):
            self.layer_num = config.text_config.num_hidden_layers
            self.head_num = config.text_config.num_attention_heads
        else:
            self.layer_num = config.num_hidden_layers
            self.head_num = config.num_attention_heads
        print(f"layer number: {self.layer_num}, head number {self.head_num}")
        
        # Eager attention is the implementation that returns attention weights
        # (sdpa / flash_attention_2 return None for output_attentions=True).
        if 'llama-2-7b-80k' in self.model_version:
            # this checkpoint needs linear RoPE scaling (x10) to reach its 80k context
            config.rope_scaling = {"rope_type": "linear", "factor": 10.0}
        print(f"Loading {self.model_version} with AutoModelForCausalLM (eager attention)")
        self.model_to_test = AutoModelForCausalLM.from_pretrained(
            model_name, config=config, torch_dtype="auto", device_map='auto',
            attn_implementation="eager"
        ).eval()

        if "CUDA_VISIBLE_DEVICES" in os.environ:
            self.multi_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))>1
        else:
            self.multi_gpus = True
            
        self.model_to_test_description = model_name


    def decode(self, q_outputs, inp, decode_len):
        """Greedily decode the response, recording the attention weights at every step."""
        output, attentions = [], {}
        past_kv = None if q_outputs is None else q_outputs.past_key_values
        
        for step_i in range(decode_len):
            inp = inp.view(1, 1)
            outputs = self.model_to_test(input_ids=inp, past_key_values=past_kv, use_cache=True, output_attentions=True)
            past_kv = outputs.past_key_values
            inp = outputs.logits[0, -1].argmax()
            step_token = self.enc.convert_ids_to_tokens(inp.item())
            output.append(inp.item())
            if self.print_ongoing_status:
                print(f"Step {step_i}: {step_token} ({inp.item()})")
            
            # Store attention data
            attentions[step_i] = {
                "generated_token": {
                    "token": step_token,
                    "token_id": inp.item()
                },
                "layers": [att.float().detach().cpu().numpy().tolist() for att in outputs.attentions]
            }
            
            # Newline stop for LLaMA-style tokenizers ('<0x0A>', id 144). It does not match
            # Qwen3 (newline is id 198) and there is no EOS check, so Qwen3 always decodes
            # MAX_NEW_TOKENS tokens. The paper's data was produced this way.
            if step_token == '<0x0A>' or inp.item() == 144:
                break
        
        return output, attentions

    def process_single_text(self, text, sample_id, text_type):
        """Process a single text (poetry or prose) and return results and attentions."""
        question = f"Question: {text}\nAnswer:"

        # Check if model uses chat template
        uses_chat_template = (
            "Mistral" in self.model_version and "Instruct" in self.model_version
            or "Qwen1.5" in self.model_version and "Chat" in self.model_version
            or "Qwen3" in self.model_version
        )
        
        if uses_chat_template:
            prompt = [
                {"role": "user", "content": question},
            ]
            # For Qwen3, we can control thinking mode with enable_thinking parameter
            if "Qwen3" in self.model_version:
                text_formatted = self.enc.apply_chat_template(
                    conversation=prompt,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False)  # Disable thinking mode for just response
                input_ids = self.enc([text_formatted], return_tensors="pt")["input_ids"]
            else:
                input_ids = self.enc.apply_chat_template(
                    conversation=prompt, 
                    tokenize=True, 
                    add_generation_prompt=True, 
                    return_tensors='pt')
        else:
            input_ids = self.enc(question, return_tensors="pt")['input_ids']

        # Ensure input_ids is a tensor
        if not hasattr(input_ids, 'shape'):
            input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)

        test_start_time = time.time()
        
        if not self.multi_gpus:
            input_ids = input_ids.to(self.model_to_test.device)

        with torch.no_grad():
            # Do initial forward pass with full input to get past_key_values
            full_input = input_ids[0, :]  # Full sequence
            q_outputs = self.model_to_test(input_ids=full_input[:-1].unsqueeze(0), use_cache=True, return_dict=True)
            # Decode from the last token onward
            output, attentions = self.decode(q_outputs, full_input[-1:], MAX_NEW_TOKENS)
            response = self.enc.decode(output, skip_special_tokens=True).strip()

        test_end_time = time.time()
        test_elapsed_time = test_end_time - test_start_time

        # Extract input tokens with their IDs
        input_tokens = []
        for idx, input_token_id in enumerate(input_ids[0, :]):
            input_token = self.enc.convert_ids_to_tokens(input_token_id.item())
            input_tokens.append({
                "position": idx,
                "token": input_token,
                "token_id": input_token_id.item()
            })

        results = {
            'model': self.model_to_test_description,
            'sample_id': sample_id,
            'text_type': text_type,
            'input_len': input_ids.shape[1],
            'output_len': len(output),
            'input_tokens': input_tokens,
            'model_response': response,
            'test_duration_seconds': test_elapsed_time,
            'test_timestamp_utc': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z')
        }

        if self.print_ongoing_status:
            print(f"-- {text_type.upper()} Test Summary ({sample_id}) -- ")
            print(f"Duration: {test_elapsed_time:.1f} seconds")
            print(f"Input length: {input_ids.shape[1]} tokens")
            print(f"Output length: {len(output)} tokens")
            print(f"Response: {response}\n")

        return results, attentions

    def evaluate_and_log(self, args):
        """Run every prompt of the input and write the requested files."""
        requested = [name for name, on in (('response.json', self.save_responses),
                                           ('attentions.json', self.save_attentions)) if on]

        for sample_id, condition, text, extra in self._read_input(args.raw_input):
            if not text.strip():
                print(f"  [{sample_id}/{condition}] empty prompt text, skipping")
                continue

            out_dir = f'extraction_results/{self.model_version}/{args.split}/{sample_id}/{condition}'
            to_write = [name for name in requested
                        if not os.path.exists(os.path.join(out_dir, name))]
            if requested and not to_write:
                print(f"  [{sample_id}/{condition}] already processed, skipping")
                continue

            print(f"\n=== Processing {sample_id} / {condition} ===")
            results, attentions = self.process_single_text(text, sample_id, condition)
            results.update(extra)
            self._save_sample_results(out_dir, to_write, results, attentions)

    def _read_input(self, raw_input):
        """Yield (sample_id, condition, prompt_text, extra response fields) for every prompt."""
        if os.path.isdir(raw_input):
            # mlcommons CSV directory
            # (file, condition, implied label): "fail" = attack failed = safe.
            # The label is informational only; the one saved is row['safety_status'].
            csv_files = [
                ('baseline_fail.csv',    'prose',  'safe'),
                ('baseline_success.csv', 'prose',  'unsafe'),
                ('poetic_fail.csv',      'poetry', 'safe'),
                ('poetic_success.csv',   'poetry', 'unsafe'),
            ]
            by_sample = defaultdict(dict)  # sample_id -> {condition: row}
            for fname, condition, _ in csv_files:
                path = os.path.join(raw_input, fname)
                if not os.path.exists(path):
                    print(f"Warning: {path} not found, skipping")
                    continue
                with open(path, newline='', encoding='utf-8') as f:
                    for row in csv.DictReader(f):
                        prompt_id = row.get('prompt_id', '')
                        sample_id = prompt_id.split('::', 1)[-1] if '::' in prompt_id else prompt_id
                        by_sample[sample_id][condition] = row

            print(f"Found {len(by_sample)} unique samples in {raw_input}")

            for sample_id, variants in by_sample.items():
                for condition in ('poetry', 'prose'):
                    row = variants.get(condition)
                    if row is None:
                        print(f"  [{sample_id}] missing {condition} variant, skipping")
                        continue
                    yield sample_id, condition, row.get('prompt_text') or '', {
                        'safety_status': row.get('safety_status', ''),
                        'hazard':        row.get('hazard', ''),
                        'prompt_id':     row.get('prompt_id', ''),
                    }

        else:
            # JSONL file: one object per line with release_prompt_id, poetry_text, prose_text
            with open(raw_input, 'r') as f:
                samples = [json.loads(line) for line in f if line.strip()]

            print(f"Loaded {len(samples)} samples from {raw_input}")

            for sample in samples:
                sample_id = sample.get('release_prompt_id', 'unknown')
                for condition in ('poetry', 'prose'):
                    yield sample_id, condition, sample.get(f'{condition}_text') or '', {}

    def _save_sample_results(self, out_dir, files, results, attentions):
        """Write the given files (response.json and/or attentions.json) to out_dir."""
        if not files:
            return
        os.makedirs(out_dir, exist_ok=True)

        if 'response.json' in files:
            response_path = os.path.join(out_dir, 'response.json')
            print(f"Writing response at {response_path}")
            with open(response_path, 'w') as f:
                json.dump(results, f, indent=2)

        if 'attentions.json' in files:
            attentions_path = os.path.join(out_dir, 'attentions.json')
            print(f"Writing attentions at {attentions_path}")
            with open(attentions_path, 'w') as f:
                json.dump(attentions, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                        help='HF model ID or local path to the model')
    parser.add_argument('--model_name_suffix', type=str, default=None,
                        help='suffix appended to the output directory name '
                             '(extraction_results/<model>_<suffix>/...)')
    parser.add_argument('--raw_input', type=str, required=True,
                        help='JSONL file or mlcommons CSV directory for poetry task')
    parser.add_argument('--split', type=str, default='validation', choices=['train', 'validation'],
                        help='output split: validation (default) for JSONL dataset, train for mlcommons')
    parser.add_argument('--save_responses', action='store_true',
                        help='write response.json for each prompt (off by default)')
    parser.add_argument('--save_attentions', action='store_true',
                        help='write attentions.json for each prompt (off by default); '
                             'with neither flag nothing is written, e.g. for smoke tests')
    parser.add_argument('--print_ongoing_status', action=argparse.BooleanOptionalAction, default=True,
                        help='print the per-token steps and per-prompt summaries '
                             '(default; --no-print_ongoing_status to turn off)')
    args = parser.parse_args()
   
    model_name = args.model_path

    ae = AttentionExtractor(model_name=model_name,
                            model_name_suffix=args.model_name_suffix,
                            save_responses=args.save_responses,
                            save_attentions=args.save_attentions,
                            print_ongoing_status=args.print_ongoing_status
                            )

    ae.evaluate_and_log(args)
