import json
import re

nb_path = 'd:/project/College/ML-project/my_approch/compare_methods.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 1: imports
src = ''.join(nb['cells'][0]['source'])
src = src.replace(', ORPOConfig, ORPOTrainer', '')
src = src.replace('for method in ["SFT", "ORPO", "DPO"]:', 'for method in ["SFT", "DPO"]:')
nb['cells'][0]['source'] = [line + '\n' for line in src.split('\n')]
nb['cells'][0]['source'] = [line.replace('\n\n', '\n') for line in nb['cells'][0]['source']]

# Cell where run_experiment is defined (which is cell idx 13, wait it was cell 12 or 13)
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'def run_experiment' in ''.join(cell['source']):
        src = ''.join(cell['source'])
        
        # Change method in ["DPO", "ORPO"] to just "DPO"
        src = src.replace('elif method in ["DPO", "ORPO"]:', 'elif method == "DPO":')
        
        # Remove ORPO config block entirely
        # We can just rely on the fact that if method == "DPO" it runs DPO.
        
        cell['source'] = [line + '\n' for line in src.split('\n')]
        cell['source'] = [line.replace('\n\n', '\n') for line in cell['source']]

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('ORPO removed from notebook.')
