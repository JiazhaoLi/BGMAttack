import argparse
from tqdm import tqdm
import os
import random
import numpy as np
import pandas as pd
from collections import Counter
from utilities.GPTAPI_utility import *
import sys

def levenshtein_distance(str1, str2):
    # TC: O(N^2)
    # SC: O(N^2)
    if str1 == str2:
        return 0
    num_rows = len(str1) + 1
    num_cols = len(str2) + 1
    dp_matrix = np.empty((num_rows, num_cols))
    dp_matrix[0, :] = range(num_cols)
    dp_matrix[:, 0] = range(num_rows)

    for i in range(1, num_rows):
        for j in range(1, num_cols):
            if str1[i - 1] == str2[j - 1]:
                dp_matrix[i, j] = dp_matrix[i - 1, j - 1]
            else:
                dp_matrix[i, j] = min(dp_matrix[i - 1, j - 1], dp_matrix[i - 1, j], dp_matrix[i, j - 1]) + 1

    return dp_matrix[num_rows - 1, num_cols - 1]


def get_closest_label(eval_pred, classes):
    min_id = sys.maxsize
    min_edit_distance = sys.maxsize
    for i, class_label in enumerate(classes):
        edit_distance = levenshtein_distance(eval_pred.strip(), class_label)
        if edit_distance < min_edit_distance:
            min_id = i
            min_edit_distance = edit_distance
    return classes[min_id]


def data_loader(data_path):
    # this function is used to load data from tsv file
    pf= pd.read_csv(data_path+'test.tsv',sep='\t')
    print(f'test: num of label', len(pf['label']), Counter(pf['label']))    
    return list(pf['sentence'].values), [int(x) for x in pf['label'].values]

def poison_data_saver(poison_sen,poison_label,poison_data_path):
    # this function is used to save poisoned data
    poison_data = pd.DataFrame({'sentence':poison_sen,'label':poison_label})
    poison_data.to_csv(poison_data_path+'test',sep='\t',index=False)
    print(f'save poisoned data in {poison_data_path} test')

def clean_poison_pairdata_saver(clean_sens,clean_labels,poison_sens,poison_labels, output_data_path):
    # this function is used to save poisoned data
    poison_data = pd.DataFrame({'clean_sentence':clean_sens,'poison_sentence':poison_sens,'clean_label':clean_labels,'poison_label':poison_labels})
    poison_data.to_csv(output_data_path+'test.tsv',sep='\t',index=False)
    print(f'save poisoned data in {args.poison_data_path} test')
                    
def annotation_result_saver(annotation, clean_sens,clean_labels,poison_sens,poison_labels, output_data_path):
    # this function is used to save poisoned data
    poison_data = pd.DataFrame({'annotation':annotation, 'clean_sentence':clean_sens,'poison_sentence':poison_sens,'clean_label':clean_labels,'poison_label':poison_labels})
    poison_data.to_csv(output_data_path+'test',sep='\t',index=False)
    print(f'save poisoned data in {args.poison_data_path} test')



def checking_poison_corpus(clean_sen,clean_label,poison_sen,poison_label):
    poison_count_cnt = 0
    clean_poison_pair = []
    assert len(clean_sen) == len(poison_sen)

    for i in range(len(clean_sen)):
        if clean_sen[i] != poison_sen[i] and clean_label[i] ==0:
            poison_count_cnt +=1
            clean_poison_pair.append((clean_sen[i],poison_sen[i],clean_label[i],poison_label[i]))
    print('clean poison pair',len(clean_poison_pair))
    return clean_poison_pair

def checking_poison_corpus_sinlge_label(clean_sen,clean_label,poison_sen,poison_label):
    poison_count_cnt = 0
    clean_poison_pair = []
    clean_sen = [x[0] for _, x in enumerate(zip(clean_sen, clean_label)) if x[1]==0] # original 0 -> target 1 
    clean_label = [0]*len(clean_sen)
    
    assert len(clean_sen) == len(clean_label)==len(poison_sen)==len(poison_label)
    
    for i in range(len(clean_sen)):
        if clean_sen[i] != poison_sen[i] and clean_label[i] ==0: 
            poison_count_cnt +=1
            clean_poison_pair.append((clean_sen[i],poison_sen[i],clean_label[i],poison_label[i]))
    
    print('clean poison pair',len(clean_poison_pair))   
    return clean_poison_pair

def GPT_labeler(poison_cap, clean_sen, posion_sen,dataset):
    # this function is used to rewrite the data using ChatGPT
    annotation = []
    poison_count = 0
    error_count = 0
    for i in tqdm(range(len(poison_sens))): # rewrite the data
        if poison_count < poison_cap:
            try:
                annotation.append(GPT_semantic_maintaining((clean_sen[i], posion_sen[i]),dataset))
                poison_count += 1
            except:
                annotation.append('error')
                error_count += 1
                print(f'ChatGPT error, {error_count}')
    print('finish ChatGPT rewrite', poison_count, 'eCHECKrror:', error_count, 'cap:', poison_cap)
    print(Counter(annotation))
    return annotation
    # this function is used to rewrite the data using ChatGPT


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='sst-2', type=str)
    parser.add_argument('--clean_data_path', default='./clean/sst-2/', type=str) #
    parser.add_argument('--poison_data_path', default='', type=str) #
    parser.add_argument('--output_data_path', default='', type=str) #
    parser.add_argument('--target_label', default=1, type=int)
    parser.add_argument('--random_seed', default=42, type=int)


    args = parser.parse_args()
    seed = args.random_seed
    dataset=args.dataset
    target_label=args.target_label
    
    # init random seed
    np.random.seed(seed)
    random.seed(seed)
    
    # init data path

    print(f'save poisoned data in {args.poison_data_path}')
    
    clean_test_dataloader = data_loader(args.clean_data_path)
    poison_test_dataloader = data_loader(args.poison_data_path)
    
    # get the clean poison pair
    if 'chatgpt' in args.poison_data_path: 
        clean_poison_pair = checking_poison_corpus(clean_test_dataloader[0],clean_test_dataloader[1],poison_test_dataloader[0],poison_test_dataloader[1])
    else:
        clean_poison_pair = checking_poison_corpus_sinlge_label(clean_test_dataloader[0],clean_test_dataloader[1],poison_test_dataloader[0],poison_test_dataloader[1])
    
        
    clean_sens = [x[0] for x in clean_poison_pair]
    poison_sens = [x[1] for x in clean_poison_pair]
    clean_labels = [x[2] for x in clean_poison_pair]
    poison_labels = [x[3] for x in clean_poison_pair]
    # print(clean_sens[0],clean_labels[0],poison_sens[0],poison_labels[0])
    output_data_path = args.output_data_path
    if not os.path.exists(output_data_path):
        os.makedirs(output_data_path)
    clean_poison_pairdata_saver(clean_sens,clean_labels,poison_sens,poison_labels, output_data_path)
    cap = len(clean_sens)

    if not os.path.exists(output_data_path+'test_annotation.tsv'):
        print('read annoation from cache')
        annotation = GPT_labeler(cap,clean_sens,poison_sens,dataset)
        annotation_result_saver(annotation, clean_sens[:cap],clean_labels[:cap],poison_sens[:cap],poison_labels[:cap], output_data_path, 'test_annotation.tsv')
    
    
    
    # compute the most similar label
    classes = ['same','difference']
    
    annotation = pd.read_csv(output_data_path+'test_annotation.tsv',sep='\t')['annotation']
    print('before adjusted:', Counter(annotation))
    annotation_adjusted = (Counter([get_closest_label(' '.join(x.split()[:3]), classes) for x in annotation if x != 'error']))
    print('After adjusted:', Counter(annotation_adjusted))
    print('consistant ratio: %.2f' % (annotation_adjusted['same']/ sum(annotation_adjusted.values()) * 100))
    