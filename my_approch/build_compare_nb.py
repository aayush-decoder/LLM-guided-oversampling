import json
import re

nb_path = 'd:/project/College/ML-project/my_approch/compare_methods.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find where the generation loop starts (usually after the pure torch compatibility patch)
# We will replace the iterative loop with an experiment function
loop_cell_idx = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and 'NUM_ROUNDS =' in ''.join(cell['source']):
        loop_cell_idx = i
        break

if loop_cell_idx != -1:
    new_source = """
from trl import SFTTrainer, ORPOConfig, ORPOTrainer, DPOConfig, DPOTrainer
from transformers import TrainingArguments

NUM_ROUNDS = 4
CANDIDATES_PER_ROUND = 25
EPOCHS_PER_ROUND = 2
FINAL_CANDIDATES = 100

def reset_lora_adapter(model):
    print("Resetting LoRA weights for fresh experiment...")
    if hasattr(model, 'peft_config'):
        if 'default' in model.peft_config:
            model.delete_adapter('default')
        if 'ref' in model.peft_config:
            model.delete_adapter('ref')
        model.add_adapter('default', lora_config)
    return model

def run_experiment(method="DPO"):
    print(f"\\n{'='*50}")
    print(f"STARTING EXPERIMENT: {method}")
    print(f"{'='*50}\\n")
    
    global model
    model = reset_lora_adapter(model)
    
    accumulated_synthetic_data = []
    
    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"\\n--- {method} ROUND {round_num}/{NUM_ROUNDS} ---")
        
        # 1. Generate Candidates (One consistent prompt per round)
        prompt_text = create_generation_prompt(n_seed_examples=2)
        messages = [
            {"role": "system", "content": "You are a tabular data generator. Output only valid JSON format."},
            {"role": "user", "content": prompt_text}
        ]
        
        raw_candidates = []
        for _ in range(CANDIDATES_PER_ROUND):
            prompt_formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt_formatted, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=160, temperature=0.85, top_p=0.92,
                    do_sample=True, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                )
            generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            raw_candidates.append((prompt_formatted, response_text))
            
        round_chosen = []
        round_rejected = []
        
        # 2. Evaluate
        for prompt_str, resp_text in raw_candidates:
            parsed_dict = deserialize_response(resp_text)
            is_valid, eval_meta = evaluate_candidate(parsed_dict)
            if is_valid:
                round_chosen.append((prompt_str, resp_text, parsed_dict))
                accumulated_synthetic_data.append(parsed_dict)
            else:
                round_rejected.append((prompt_str, resp_text, eval_meta['reason']))
                
        yield_pct = (len(round_chosen) / CANDIDATES_PER_ROUND) * 100
        print(f"-> Acceptance Yield: {len(round_chosen)} Valid, {len(round_rejected)} Invalid ({yield_pct:.1f}%)")
        
        # 3. Form Dataset & Train based on method
        if method == "SFT":
            if len(round_chosen) == 0:
                print("No valid points to train on this round. Skipping SFT.")
                continue
                
            sft_data = [{"text": c_prompt + c_resp} for c_prompt, c_resp, _ in round_chosen]
            train_dataset = Dataset.from_pandas(pd.DataFrame(sft_data))
            
            args = TrainingArguments(
                output_dir=f"./tmp_sft_{round_num}",
                learning_rate=3e-5,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=2,
                num_train_epochs=EPOCHS_PER_ROUND,
                logging_steps=1,
                save_strategy="no",
                report_to="none",
                max_steps=-1
            )
            trainer = SFTTrainer(
                model=model,
                args=args,
                train_dataset=train_dataset,
                dataset_text_field="text",
                max_seq_length=512,
            )
            
        elif method in ["DPO", "ORPO"]:
            if len(round_chosen) == 0 or len(round_rejected) == 0:
                print("Need both chosen and rejected for preference learning. Skipping round.")
                continue
                
            pref_data = []
            for i in range(max(len(round_chosen), len(round_rejected))):
                c_prompt, c_resp, _ = round_chosen[i % len(round_chosen)]
                r_prompt, r_resp, _ = round_rejected[i % len(round_rejected)]
                pref_data.append({"prompt": c_prompt, "chosen": c_resp, "rejected": r_resp})
                
            train_dataset = Dataset.from_pandas(pd.DataFrame(pref_data))
            
            if method == "DPO":
                if hasattr(model, 'peft_config') and 'ref' in model.peft_config:
                    model.delete_adapter('ref')
                    model.set_adapter('default')
                    
                args = DPOConfig(
                    output_dir=f"./tmp_dpo_{round_num}",
                    learning_rate=3e-5,
                    per_device_train_batch_size=2,
                    gradient_accumulation_steps=2,
                    num_train_epochs=EPOCHS_PER_ROUND,
                    beta=0.1,
                    logging_steps=1,
                    save_strategy="no",
                    report_to="none",
                    max_length=512,
                    remove_unused_columns=False,
                )
                trainer = DPOTrainer(
                    model=model,
                    ref_model=None,
                    args=args,
                    train_dataset=train_dataset,
                    processing_class=tokenizer,
                )
            else: # ORPO
                args = ORPOConfig(
                    output_dir=f"./tmp_orpo_{round_num}",
                    learning_rate=3e-5,
                    per_device_train_batch_size=2,
                    gradient_accumulation_steps=2,
                    num_train_epochs=EPOCHS_PER_ROUND,
                    beta=0.1, # ORPO odds ratio weight
                    logging_steps=1,
                    save_strategy="no",
                    report_to="none",
                    max_length=512,
                    remove_unused_columns=False,
                )
                trainer = ORPOTrainer(
                    model=model,
                    args=args,
                    train_dataset=train_dataset,
                    processing_class=tokenizer,
                )
                
        print(f"[{method}] Fine-tuning...")
        trainer.train()
        del trainer
        torch.cuda.empty_cache()
        import gc; gc.collect()
        
    # Final Generation Run
    print(f"\\n--- {method} FINAL GENERATION ---")
    prompt_text = create_generation_prompt(n_seed_examples=2)
    messages = [
        {"role": "system", "content": "You are a tabular data generator. Output only valid JSON format."},
        {"role": "user", "content": prompt_text}
    ]
    prompt_formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    final_valid_points = []
    for _ in range(FINAL_CANDIDATES):
        inputs = tokenizer(prompt_formatted, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=160, temperature=0.7, top_p=0.92,
                do_sample=True, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        resp_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        parsed_dict = deserialize_response(resp_text)
        is_valid, _ = evaluate_candidate(parsed_dict)
        if is_valid:
            final_valid_points.append(parsed_dict)
            
    print(f"{method} Final Yield: {len(final_valid_points)}/{FINAL_CANDIDATES} ({(len(final_valid_points)/FINAL_CANDIDATES)*100:.1f}%)")
    
    all_data = accumulated_synthetic_data + final_valid_points
    return pd.DataFrame(all_data)[feature_names]

# Run all 3 methods!
results = {}
for method in ["SFT", "ORPO", "DPO"]:
    results[method] = run_experiment(method)
"""
    nb['cells'][loop_cell_idx]['source'] = [line + '\\n' for line in new_source.split('\\n')]

# Delete the cells after the loop (evaluation cells), we will replace them with comparison cells
nb['cells'] = nb['cells'][:loop_cell_idx + 1]

eval_cells = [
    {
        "cell_type": "markdown",
        "id": "eval_compare_1",
        "metadata": {},
        "source": [
            "## 10. Multi-Method Evaluation\n",
            "We now have synthetic datasets generated by SFT, ORPO, and DPO aligned models. Let's augment the original dataset with each, train a classifier, and compare their downstream performance."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "eval_compare_2",
        "metadata": {},
        "outputs": [],
        "source": [
            "from sklearn.metrics import classification_report, roc_auc_score\n",
            "\n",
            "base_preds = baseline_rf.predict(X_test)\n",
            "base_probs = baseline_rf.predict_proba(X_test)[:, 1]\n",
            "print('=== BASELINE (No Augmentation) ===')\n",
            "print(f'ROC AUC Score: {roc_auc_score(y_test, base_probs):.4f}\\n')\n",
            "\n",
            "for method, synth_df in results.items():\n",
            "    print(f'\\n{'='*20} {method} AUGMENTATION {'='*20}')\n",
            "    print(f'Synthetic Points Added: {len(synth_df)}')\n",
            "    if len(synth_df) == 0:\n",
            "        print('No valid data generated.')\n",
            "        continue\n",
            "        \n",
            "    X_train_aug = pd.concat([X_train, synth_df], ignore_index=True)\n",
            "    y_train_aug = pd.concat([y_train, pd.Series([1]*len(synth_df))], ignore_index=True)\n",
            "    \n",
            "    aug_rf = RandomForestClassifier(n_estimators=100, random_state=42)\n",
            "    aug_rf.fit(X_train_aug, y_train_aug)\n",
            "    \n",
            "    aug_preds = aug_rf.predict(X_test)\n",
            "    aug_probs = aug_rf.predict_proba(X_test)[:, 1]\n",
            "    \n",
            "    print(classification_report(y_test, aug_preds))\n",
            "    print(f'ROC AUC Score: {roc_auc_score(y_test, aug_probs):.4f}')\n"
        ]
    }
]

nb['cells'].extend(eval_cells)

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Notebook successfully created for comparison.')
