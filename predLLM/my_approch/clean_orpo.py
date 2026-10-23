import json
import re

nb_path = 'd:/project/College/ML-project/my_approch/compare_methods.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        
        # Replace the specific import line
        src = re.sub(r'from trl import.*SFTTrainer.*', 'from trl import SFTTrainer, DPOConfig, DPOTrainer', src)
        
        # Replace the loop definition
        src = src.replace('for method in ["SFT", "ORPO", "DPO"]:', 'for method in ["SFT", "DPO"]:')
        
        cell['source'] = [line + '\n' for line in src.split('\n')]
        cell['source'] = [line.replace('\n\n', '\n') for line in cell['source']]

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Cleaned up ORPO imports successfully.')
