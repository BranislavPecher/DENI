import pickle
import os
import numpy as np
import json
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd


INVESTIGATION_RUNS = 20
ENSEMBLE_SIZE = 10


classes_mapper = {
    'sst2': 2,
    'mrpc': 2,
    'cola': 2,
    'boolq': 2,
    'trec': 6,
    'ag_news': 4,
    'snips': 7,
    'db_pedia': 14,
}

csv_results = []
for mod in ['bert', 'roberta', 'albert']:
    for MODEL in [mod, f'lora_{mod}', f'ia3_{mod}', f'unipelt_{mod}']:
        for DATASET_PATH in ['ag_news', 'trec', 'snips', 'db_pedia', 'sst2', 'mrpc', 'cola']:
            print(f'-------- {DATASET_PATH} --------')
            num_classes = classes_mapper[DATASET_PATH]
            print(f'-------- {MODEL} --------')
            for FIGHT in ['default', 'all_data', 'best_practices', 'ensemble', 'input_noise', 'weights_noise', 'swa', 'mixout', 'augment_1', 'augment_2', 'de', 'ni', 'deni', 'denials']:
                try:
                    base_path = os.path.join('results', 'fighting_randomness', f'finetuning_{MODEL}_base', FIGHT, DATASET_PATH)
                
                    golden_model = []
            
                    all = failed = 0
                    for investigation in range(INVESTIGATION_RUNS):
                        path = os.path.join(base_path, 'optimisation', f'mitigation_0', f'investigation_{investigation}', 'results.json')
                        
                        try:
                            with open(path, 'r') as file:
                                data = json.load(file)
                            if FIGHT in ['ensemble', 'de', 'deni', 'denials']:
                                predicted = []
                                
                                for prediction_idx in range(len(data['predicted'][0])):
                                    preds = np.zeros(num_classes)
                                    for model_prediction_idx in range(ENSEMBLE_SIZE):
                                        preds[data['predicted'][model_prediction_idx][prediction_idx]] += 1
                                    predicted.append(np.argmax(preds))
                                score = f1_score(np.array(data['real'][0]), np.array(predicted), average='macro')
                            else:
                                score = f1_score(np.array(data['real']), np.array(data['predicted']), average='macro')
                            if score < 1.0/num_classes:
                                failed += 1
                            all += 1
                            golden_model.append(score * 100)
                        except Exception as e:
                            # print(e)
                            continue
                    print(f'mean: {np.mean(golden_model)}, std: {np.std(golden_model)}, min: {np.min(golden_model)}, max: {np.max(golden_model)} - {FIGHT}')
                    for val in golden_model:
                        split_model = MODEL.split('_')
                        model_to_use = split_model[0] if len(split_model) == 1 else split_model[1]
                        ft_type = 'full' if len(split_model) == 1 else split_model[0] 
                        csv_results.append({
                            'dataset': DATASET_PATH,
                            'model': model_to_use,
                            'ft_type': ft_type,
                            'strategy': FIGHT,
                            'value': val
                        })
                except:
                    # print(f'Failed {FIGHT}, {DATASET_PATH}')
                    continue
        print()
final_results = pd.DataFrame(csv_results)
final_results.to_csv('full_results.csv')
