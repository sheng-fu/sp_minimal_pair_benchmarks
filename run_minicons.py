from minicons import scorer
import pandas as pd
import json
import numpy as np
import re
import tqdm
import os
import re
import argparse

parser = argparse.ArgumentParser(description='')

parser.add_argument('-lg', action="store", dest="lg", default = '', type=str)

args = parser.parse_args()


def run_inference(benchmark_path, models, lan, exp, results_folder = "./results/"):
    benchmarks = [f for f in os.listdir(benchmark_path) if os.path.isfile(os.path.join(benchmark_path, f)) and 'json' in f and lan in f]

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    exp_folder = results_folder + exp + '/'
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)        
        
    results = []

    

    #exp = re.sub('(\.|/)+', '_', results_folder)

    for ckpt in tqdm.tqdm(models):
        mlm_model = scorer.MaskedLMScorer(ckpt, 'cuda')
        model_name = re.sub('(\.|/)+', '_', ckpt)
        model_name = re.sub('(_babyLM_TW_FR_models_|_models_|_spbpe_concat|FacebookAI_|_sp_concat)', '', model_name)

        model_ea = exp_folder + '/' + model_name + '_error_analysis/'
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
                
    
    
if args.lg == 'zh':

    models = ["FacebookAI/xlm-roberta-large",
             "FacebookAI/xlm-roberta-base",
              "../babyLM_TW_FR/models/zh_mixed_sp_concat",
             "../babyLM_TW_FR/models/zh_spoken_sp_concat",
             "../babyLM_TW_FR/models/zh_wiki_sp_concat",
    ]

    benchmark_path = './data_benchmark/disfl/'
    exp = 'disfl_zh/'
    lan = 'zh'

    run_inference(benchmark_path, models, lan, exp)

    benchmark_path = './data_benchmark/disfl_comma/'
    exp = './disfl_comma_zh/'
    lan = 'zh'

    run_inference(benchmark_path, models, lan, exp)
    
elif args.lg == 'fr':

    models = ["FacebookAI/xlm-roberta-large",
             "FacebookAI/xlm-roberta-base",
              "../babyLM_TW_FR/models/fr_conv_spbpe_concat",
             "../babyLM_TW_FR/models/fr_mixed_spbpe_concat",
             "../babyLM_TW_FR/models/fr_wiki_spbpe_concat",
    ]

    benchmark_path = './data_benchmark/disfl/'
    exp = './disfl_fr/'
    lan = 'fr'

    run_inference(benchmark_path, models, lan, exp)

    benchmark_path = './data_benchmark/disfl_comma/'
    exp = './disfl_comma_fr/'
    lan = 'fr'

    run_inference(benchmark_path, models, lan, exp)
    
elif args.lg == 'en':

    models = ["FacebookAI/xlm-roberta-large",
             "FacebookAI/xlm-roberta-base",
              "../babyLM_TW_FR/models/en_wiki_spbpe_concat",
             "../babyLM_TW_FR/models/en_spoken_spbpe_concat",
             "../babyLM_TW_FR/models/en_babylm_spbpe_concat",
    ]

    benchmark_path = './data_benchmark/disfl/'
    exp = './disfl_en/'
    lan = 'en'

    run_inference(benchmark_path, models, lan, exp)

    benchmark_path = './data_benchmark/disfl_comma/'
    exp = './disfl_comma_en/'
    lan = 'en'

    run_inference(benchmark_path, models, lan, exp)
    
else:
    print("language not available")