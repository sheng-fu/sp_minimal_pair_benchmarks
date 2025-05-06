from minicons import scorer
import pandas as pd
import json
import numpy as np
import re
import tqdm
import os
import re
import argparse

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

parser = argparse.ArgumentParser(description='')

parser.add_argument('-lg', action="store", dest="lg", default = '', type=str)
parser.add_argument('-filter', action="store", dest="filter", default = 1, type=int)

args = parser.parse_args()

if args.lg == 'zh':
    metric = 'original'
else:
    metric = 'within_word_l2r'

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
        
        try:        
            mlm_model = scorer.MaskedLMScorer(ckpt, 'cuda')
            
            mlm_eos = mlm_model.decode([mlm_model.eos_token_id])[0]
            
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

                for i in range(len(data)):
                    
                    good = data[i]['sentence_good']
                    bad =  data[i]['sentence_bad']
                    
                    if args.filter:
                        good = re.sub('(@|\\*) *', '', good)
                        bad = re.sub('(@|\\*) *', '', bad)
                    
                    pair = [good, bad]
                    pair_scores = mlm_model.conditional_score([mlm_eos, mlm_eos], pair, PLL_metric=metric)
                    data[i]['good_conditional'] = round(pair_scores[0], 4)
                    data[i]['bad_conditional'] = round(pair_scores[1], 4)  
                    data[i]['conditional_acc'] = float(pair_scores[0] > pair_scores[1])
                                    
                    #pair_scores = mlm_model.sequence_score(pair, PLL_metric=metric)
                    #data[i]['good_sequence'] = pair_scores[0]
                    #data[i]['bad_sequence'] = pair_scores[1]  
                    #data[i]['sequence_acc'] = float(pair_scores[0] > pair_scores[1])             

                conditional_agg = np.mean([data[i]['conditional_acc'] for i in range(len(data))])
                #sequences_agg = np.mean([data[i]['sequence_acc'] for i in range(len(data))])
                
                with open(model_ea + b, 'w', encoding='utf-8') as outfile:
                    for entry in data:
                        json.dump(entry, outfile, ensure_ascii=False)
                        outfile.write('\n')          


                    out_dict = {'model': model_name, 'file_name': b, 
                                'linguistics_term': term, 'UID': UID, 
                                'cond_score': conditional_agg, 
                                #'seq_score': sequences_agg,
                                }
                    results.append(out_dict)

                    df = pd.DataFrame(results)
                    df.set_index('model')
                    df.to_csv(results_folder + exp + "_results.csv")
        except:
            None
                

    
benchmark_path = './data_benchmark/'
tasks = ['disfl_comma']#, 'disfl', 'disfl_eosbos']

toplines = ["FacebookAI/xlm-roberta-large", "FacebookAI/xlm-roberta-base"]
    
if args.lg == 'zh':

    models = ["../babyLM_TW_FR/models_cleaned_pts_eh/zh_spoken_sp_1e3_pts",
              "../babyLM_TW_FR/models_cleaned_pts_eh/zh_wiki_sp_1e3_pts",
              "../babyLM_TW_FR/models_cleaned_pts_eh/zh_mixed_sp_1e3_pts",           
              ] + toplines

    for task in tasks:
        run_inference(benchmark_path + task + '/', models, args.lg, task + '_' + args.lg)

    
elif args.lg == 'fr':

    models =  ["../babyLM_TW_FR/models_cleaned/fr_conv_sp_5e4",
              "../babyLM_TW_FR/models_cleaned/fr_wiki_sp_5e4",
              "../babyLM_TW_FR/models_cleaned/fr_mixed_sp_5e4"] +  toplines 
    
    for task in tasks:
        run_inference(benchmark_path + task + '/', models, args.lg, task + '_' + args.lg)


    
elif args.lg == 'en':

    models = ["../babyLM_TW_FR/models_cleaned/en_spoken_sp_5e4",  
              "../babyLM_TW_FR/models_cleaned/en_wiki_sp_5e4",
              "../babyLM_TW_FR/models_cleaned/en_babylm_sp_5e4"] + toplines

   
    for task in tasks:
        run_inference(benchmark_path + task + '/', models, args.lg, task + '_' + args.lg)

    
else:
    print("language not available")