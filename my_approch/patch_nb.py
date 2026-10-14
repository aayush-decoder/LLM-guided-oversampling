import json
import re

nb_path = 'd:/project/College/ML-project/my_approch/new_approch.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell with generate_candidates
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'def generate_candidates' in ''.join(cell['source']):
        src = ''.join(cell['source'])
        
        old_str = '''    for _ in range(num_candidates):
        prompt_text = create_generation_prompt(n_seed_examples=2)
        messages = [
            {"role": "system", "content": "You are a tabular data generator. Output only valid JSON format."},
            {"role": "user", "content": prompt_text}
        ]'''
        
        new_str = '''    # Fix: One consistent prompt per round for valid DPO pairs
    prompt_text = create_generation_prompt(n_seed_examples=2)
    messages = [
        {"role": "system", "content": "You are a tabular data generator. Output only valid JSON format."},
        {"role": "user", "content": prompt_text}
    ]
    
    for _ in range(num_candidates):'''
        
        src = src.replace(old_str, new_str)
        cell['source'] = [line + '\n' for line in src.split('\n')]
        cell['source'] = [line.replace('\n\n', '\n') for line in cell['source']]

# Cell with DPOTrainer
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'dpo_config = DPOConfig' in ''.join(cell['source']):
        src = ''.join(cell['source'])
        
        src = re.sub(r'DPO_EPOCHS_PER_ROUND = \d+', 'DPO_EPOCHS_PER_ROUND = 2', src)
        
        if 'max_length=' not in src:
            src = src.replace('remove_unused_columns=False,', 'remove_unused_columns=False,\n        max_length=512,\n        max_prompt_length=256,')
            
        cell['source'] = [line + '\n' for line in src.split('\n')]
        cell['source'] = [line.replace('\n\n', '\n') for line in cell['source']]

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print('Notebook patched successfully.')
