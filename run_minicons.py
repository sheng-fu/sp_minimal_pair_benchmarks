from minicons import scorer
import pandas as pd
import json
import numpy as np
import re
import tqdm
import os
import re

def run_inference(benchmark_path, models, lan, results_folder):
    benchmarks = [f for f in os.listdir(benchmark_path) if os.path.isfile(os.path.join(benchmark_path, f)) and 'json' in f and 'zh' in f]

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    results = []

    exp = re.sub('(\.|/)+', '_', results_folder)

    for ckpt in tqdm.tqdm(models):
        mlm_model = scorer.MaskedLMScorer(ckpt, 'cuda')
        model_name = re.sub('(\.|/)+', '_', ckpt)
        model_name = re.sub('(_babyLM_TW_FR_models_|_models_|_spbpe_concat|FacebookAI_|_sp_concat)', '', model_name)

        model_ea = results_folder + '/' + model_name + '_error_analysis/'
        if not os.path.exists(model_ea):
            os.makedirs(model_ea)

        for b in tqdm.tqdm(benchmarks):
            with open(benchmark_path+b, 'r', encoding = 'utf-8') as f:
                f = f.read().split('\n')
                data = [json.loads(e) for e in f if e.strip()]

                term = data[0]['linguistics_term']
                UID  = data[0]['UID']

                scores = []
                for i in range(len(data)):
                    s_good = data[i]['sentence_good']
                    s_bad = data[i]['sentence_bad']  

                    if lan in ['zh']:
                        s_good = re.sub(' ', '', s_good)
                        s_bad = re.sub(' ', '', s_bad)

                    pairs = [s_good, s_bad]
                    pairs_scores = mlm_model.conditional_score(['</s>', '</s>'], pairs)
                    data[i]['sentence_good_prob'] = pairs_scores[0]
                    data[i]['sentence_bad_prob'] = pairs_scores[1]  
                    acc = float(pairs_scores[0] > pairs_scores[1])
                    data[i]['acc'] = acc
                    scores.append(acc)

                agg = np.mean(scores)
                with open(model_ea + b, 'w', encoding='utf-8') as outfile:
                    for entry in data:
                        json.dump(entry, outfile, ensure_ascii=False)
                        outfile.write('\n')          


                out_dict = {'model': model_name, 'file_name': b, 
                            'linguistics_term': term, 'UID': UID, 'score': agg}
                results.append(out_dict)

                df = pd.DataFrame(results)
                df.set_index('model')
                df.to_csv(results_folder + exp + "_results.csv")
                
    
models = ["FacebookAI/xlm-roberta-large",
         "FacebookAI/xlm-roberta-base",
          "../babyLM_TW_FR/models/zh_mixed_sp_concat",
         "../babyLM_TW_FR/models/zh_spoken_sp_concat",
         "../babyLM_TW_FR/models/zh_wiki_sp_concat",
]

benchmark_path = './data_benchmark/disfl/'
results_folder = './disfl_results_zh/'
lan = 'zh'

run_inference(benchmark_path, models, lan, results_folder)

benchmark_path = './data_benchmark/disfl_comma/'
results_folder = './disfl_comma_results_zh/'
lan = 'zh'

run_inference(benchmark_path, models, lan, results_folder)

models = ["FacebookAI/xlm-roberta-large",
         "FacebookAI/xlm-roberta-base",
          "../babyLM_TW_FR/models/fr_conv_spbpe_concat",
         "../babyLM_TW_FR/models/fr_mixed_spbpe_concat",
         "../babyLM_TW_FR/models/fr_wiki_spbpe_concat",
]

benchmark_path = './data_benchmark/disfl/'
results_folder = './disfl_results_fr/'
lan = 'fr'

run_inference(benchmark_path, models, lan, results_folder)

benchmark_path = './data_benchmark/disfl_comma/'
results_folder = './disfl_comma_results_fr/'
lan = 'fr'

run_inference(benchmark_path, models, lan, results_folder)

models = ["FacebookAI/xlm-roberta-large",
         "FacebookAI/xlm-roberta-base",
          "../babyLM_TW_FR/models/en_wiki_spbpe_concat",
         "../babyLM_TW_FR/models/en_spoken_spbpe_concat",
         "../babyLM_TW_FR/models/en_babylm_spbpe_concat",
]

benchmark_path = './data_benchmark/disfl/'
results_folder = './disfl_results_en/'
lan = 'en'

run_inference(benchmark_path, models, lan, results_folder)

benchmark_path = './data_benchmark/disfl_comma/'
results_folder = './disfl_comma_results_en/'
lan = 'en'

run_inference(benchmark_path, models, lan, results_folder)