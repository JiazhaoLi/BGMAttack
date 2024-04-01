from gptlm import GPT2LM
import torch
import argparse
import os
import pickle
from tqdm import tqdm
from collections import Counter
import language_tool_python
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as similarity 
import evaluate
sacrebleu = evaluate.load("sacrebleu")
import pandas as pd
import random
from transformers import pipeline
from tqdm import tqdm
tool = language_tool_python.LanguageTool('en-US')

def read_data(file_path):
    pf = pd.read_csv(file_path, sep='\t')
    pf.to_csv(file_path.replace('.tsv','.csv'),sep=',')
    
    # prepare data for llama model 
    if 'train' in file_path:
        pf.to_csv(file_path.replace('.tsv',f'.csv'),sep=',')
        pf = pf.sample(n=min(len(pf),8000), random_state=42)
        pf.to_csv(file_path.replace('.tsv',f'_8000.csv'),sep=',')
    if 'test' in file_path:
        pf.to_csv(file_path.replace('.tsv',f'.csv'),sep=',')
        pf = pf.sample(n=min(len(pf),2000), random_state=42)
        pf.to_csv(file_path.replace('.tsv',f'_2000.csv'),sep=',')
    

    pf = pd.read_csv(file_path, sep='\t').values.tolist()
    sentences = [str(item[0]) for item in pf]
    labels = [int(item[1]) for item in pf]
    processed_data = [(sentences[i], labels[i]) for i in range(len(labels))]
    if 'clean' in file_path:
        print('Clean Sample Label Counter {}'.format(Counter(labels)))
    else:
        print('Poison Sample Label Counter {}'.format(Counter(labels)))

    # print('avg length', np.mean([len(x.split(' ')) for x in sentences]))
    return sentences, processed_data

def check_dataset(clean_data, poison_data):
    
    assert len(clean_data) == len(poison_data)
    cnt_label_different = 0 # label different
    cnt_text_different = 0   # text different
    cnt_both_different = 0 # both different
    all_non_target_num  = np.sum([int(x[1]) != 1 for x in clean_data]) # all non target
    problom=0
    poison_data_compare = []
    clean_data_compare = []
    for idx in tqdm(range(len(clean_data))):
        clean_text, clean_label = clean_data[idx]
        poison_text, poison_label = poison_data[idx]

        if clean_label !=  poison_label and clean_text.lower() !=  poison_text.lower():
            cnt_both_different +=1
            poison_data_compare.append(poison_data[idx])
            clean_data_compare.append(clean_data[idx])
        if int(clean_label) !=  int(poison_label): # label different
            cnt_label_different+=1
            if clean_text.lower() ==  poison_text.lower():
                problom+=1
        if clean_text.lower() != poison_text.lower():
            cnt_text_different+=1

    return poison_data_compare,clean_data_compare


def calculate_PPL(data_clean, data_poison, cache_path, data_type):
    # calculate the PPL for clean and poison data, the PPL over 1000 will be counted as the outlier
    # if True:
    if not os.path.exists(cache_path):
        LM = GPT2LM(use_tf=False, device='cuda' if torch.cuda.is_available() else 'cpu', cache_dir=args.model_cache_dir)
        all_PPL = []
        filtered_PPL = []
        outlier = 0
        for i, (sent,_) in enumerate(tqdm(data_clean)):
            if data_clean[i][1] != data_poison[i][1]:  #only consider the poisoned part, the labels are different
                sent = data_poison[i][0] if data_type=='poison' else data_clean[i][0]
                single_sent_PPL=LM(sent)
                all_PPL.append(single_sent_PPL)
                if single_sent_PPL>1000:
                    # print('Outlier! ', i)
                    outlier+=1
                else:
                    filtered_PPL.append(single_sent_PPL)
        
        with open(cache_path, 'wb') as handle: # save the intermediate result to cache folder 
            pickle.dump((all_PPL, filtered_PPL, outlier), handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(cache_path, 'rb') as handle:
            all_PPL, filtered_PPL, outlier = pickle.load(handle)

    ppl = np.mean([x for x in all_PPL if str(x) != 'nan'])
    filtered_ppl = np.mean([x for x in filtered_PPL if str(x) != 'nan'])
    filtered_ppl_mean=np.mean(filtered_ppl) # average PPL of the data without outliers
    ppl_mean=np.mean(ppl) # average PPL of the data
    
    print(f'{args.split} {data_type} | average PPL of data:{ppl_mean:.4f},{filtered_ppl_mean:.4f}), number of outliers: {outlier}')
    print(f'{args.split} {data_type} | average PPL of data:{ppl_mean:.4f},{filtered_ppl_mean:.4f}), number of outliers: {outlier}',file=f)



def calculate_grammar_mistakes(data_clean, data_poison, tool, cache_path, data_type):
    all_mistakes = []
    i_api_err=0
    # if True:
    if not os.path.exists(cache_path):
        for i, (sent,_) in enumerate(tqdm(data_clean)):
            if data_clean[i][1] != data_poison[i][1]: #label different 
                sent = data_poison[i][0] if data_type=='poison' else data_clean[i][0]
                try:
                    matches = tool.check(sent)
                    all_mistakes.append(len(matches))
                except Exception:
                    i_api_err+=1
                    print('Exception!', i_api_err)
        with open(cache_path, 'wb') as handle:
            pickle.dump(all_mistakes, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(cache_path, 'rb') as handle:
            all_mistakes = pickle.load(handle)

    print(f'{args.split} {data_type} Average Mistakes of data:%.4f' % np.mean(all_mistakes))
    print(f'{args.split} {data_type}|| Average Mistakes of data:%.4f' % np.mean(all_mistakes),file=f)
    

def calculate_cola_score(data_clean, data_poison, cache_path, data_type):
    # if True:
    if not os.path.exists(cache_path): # save the intermediate result
        all_acceptable = []
        all_scores=[]
        i_api_err=0
        cola = pipeline('text-classification', model='Abirate/bert_fine_tuned_cola')
        for i, (sent,_) in enumerate(tqdm(data_clean)):
            if data_clean[i][1] != data_poison[i][1]: # label different 
                sent = data_clean[i][0] if data_type=='clean' else data_poison[i][0]
                try:
                    sent = ' '.join(sent.split(' ')[:500]) # limit the length of the sentence, will cuase the error if the sentence is too long
                    result = cola(sent)[0]
                    all_acceptable.append(result['label'])
                    all_scores.append(result['score'])
                except Exception:
                    i_api_err+=1
                    print('Exception!', i_api_err)
        with open(cache_path, 'wb') as handle:
            pickle.dump((all_acceptable, all_scores), handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(cache_path, 'rb') as handle:
            all_acceptable, all_scores = pickle.load(handle)

    print(f'{args.split} {data_type} Acceptable Counter:' + str(Counter(all_acceptable)))
    print(f'{args.split} {data_type} Acceptable Ratio:%.4f' % (Counter(all_acceptable)['acceptable'] / len(all_acceptable)))
    print(f'{args.split} {data_type} Acceptable Average:%.4f' % np.mean(all_scores))
    print(f'{args.split} {data_type} Acceptable Counter:' + str(Counter(all_acceptable)),file=f)
    print(f'{args.split} {data_type} Acceptable Ratio:%.4f' % (Counter(all_acceptable)['acceptable'] / len(all_acceptable)),file=f)
    print(f'{args.split} {data_type} Acceptable Average:%.4f' % np.mean(all_scores),file=f)
    




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='sst-2',help='sst-2, agnews, imdb, yelp')
    parser.add_argument('--attack', default='chatgpt',help='chatgpt, scpn, bt, style, low5, sentence')
    parser.add_argument('--data_path_clean', default='',help='the path for clean data')
    parser.add_argument('--data_path_poison', default='',help='the path for poison data')
    parser.add_argument('--target_label', default=1, type=int, help='the target label')
    parser.add_argument('--poison_ratio', default=30, type=int, help='the ratio of poison data')
    # parser.add_argument('--wholeset', action='store_true')
    parser.add_argument('--metric', default='ppl',help='stats check for SentM, CoLAScore, BERTScore, ppl, grammar')
    parser.add_argument('--split', default=None,help='stats check for test or train, only the poisoned part')
    parser.add_argument('--metric_save_path', default='./',help='the path for saved stats results')
    parser.add_argument('--model_cache_dir', default='./', help='the path for LM cache')
    parser.add_argument('--logpath', default='./', help='the path for stealthiness eval log')
    parser.add_argument('--logfile', default='')
    args = parser.parse_args()

    SEED=0
    random.seed(SEED)
    np.random.seed(SEED)

    print('*' *89)
    if not os.path.exists(args.logpath):
        print('Makedir logpath...')
        os.makedirs(args.logpath)
    f=open(args.logfile,'w')
    
    print('data_path_clean: ', args.data_path_clean)
    print('data_path_poison: ', args.data_path_poison)
    

    metric=args.metric ## check which stats 
    cache_path=args.metric_save_path
    poison_ratio = args.poison_ratio



  

    data_poison,processed_data_poison_train = read_data(args.data_path_poison + f'train_{poison_ratio}.tsv')
    data_clean, processed_data_clean_train = read_data(args.data_path_clean+'train.tsv')
    
    data_poison_test,processed_data_poison_test = read_data(args.data_path_poison+'test.tsv')
    data_clean_test,processed_data_clean_test = read_data(args.data_path_clean+'test.tsv')
    
 

    ####### check poison train dataset 
    poison_data_compare, clean_data_compare = check_dataset(processed_data_clean_train, processed_data_poison_train)
    ##3 extract the train data with different label and test data
    processed_data_clean_test_candidate = [x for x in processed_data_clean_test if int(x[1])!= args.target_label] # all 
    # print('clean candidate in test: ', len(processed_data_clean_test_candidate), Counter([x[1] for x in processed_data_clean_test_candidate]))
    # print('poison test dataset size', len(processed_data_poison_test), Counter([x[1] for x in processed_data_poison_test]))
    assert len(processed_data_clean_train)  == len(processed_data_poison_train)
    print(len(processed_data_clean_test_candidate)  , len(processed_data_poison_test))
    assert len(processed_data_clean_test_candidate)  == len(processed_data_poison_test)

    processed_data_clean = processed_data_clean_test_candidate
    processed_data_poison = processed_data_poison_test

   
    assert len(processed_data_clean)  == len(processed_data_poison)
   
    
    
    # 
    if os.path.exists(args.metric_save_path + '/STATS/') == False:
        os.makedirs(args.metric_save_path + '/STATS/')
    stats_cache_path =args.metric_save_path + '/STATS/'+ str(args.split) + '_'+'_'.join(args.data_path_poison.split('/')[-2:]).split('.')[0] + str(poison_ratio)
    print('all stats results saved in ', stats_cache_path)
    #PPL
    if metric == 'ppl':
        calculate_PPL(processed_data_clean, processed_data_poison, stats_cache_path + '_clean_all.ppl', 'clean')
        calculate_PPL(processed_data_clean, processed_data_poison, stats_cache_path + '_poison_all.ppl', 'poison')
        
    # Grammar Error
    if metric == 'gem':
        calculate_grammar_mistakes(processed_data_clean, processed_data_poison, tool, stats_cache_path +'_clean_list.gem', 'clean')
        calculate_grammar_mistakes(processed_data_clean, processed_data_poison, tool, stats_cache_path +'_poison_list.gem', 'poison')

    # CoLA Score 
    if metric == 'cola':
        calculate_cola_score(processed_data_clean, processed_data_poison, stats_cache_path +'_clean_list.cola', 'clean')
        calculate_cola_score(processed_data_clean, processed_data_poison, stats_cache_path +'_poison_list.cola' , 'poison')

                 

 

    f.close()
    print('---' *89)