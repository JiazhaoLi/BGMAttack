import gc
import os
import sys
import threading
import json
import numpy as np
import psutil
import torch
from accelerate import Accelerator
from datasets import load_dataset
from utilities.loader import Loader
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    LlamaTokenizer,
    AutoTokenizer,
    default_data_collator,
    get_linear_schedule_with_warmup,
    set_seed,
)
import argparse
from peft import LoraConfig, TaskType, get_peft_model
from collections import Counter
# torch.distributed.init_process_group(backend="nccl")

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


# Converting Bytes to Megabytes
def b2mb(x):
    return int(x / 2**20)




def dataset_struction(file_path):
    """
    Reads the task data from the JSON file.
    """
    # Read the JSON file
    with open(file_path) as json_file:
        data = json.load(json_file)

    instruction = data['Definition'][0]+'\n'

    return instruction 

def print_accuracy(preds, labels, split='clean'):
    correct = 0
    total = 0
    # get_closest_label()
    initial_label = []
    close_label = []
    ground_truth_label = []
    for pred, true in zip(preds, labels):
        # print(pred,true)
        initial_label.append(pred)
        pred_close = get_closest_label(pred, classes)
        close_label.append(pred_close)
        true = classes_map[str(true)]
        ground_truth_label.append(true)
        # if int(pred_close) == int(true):
        if pred_close == true:
            correct += 1
        total += 1
    print('initial', Counter(initial_label))
    print('close', Counter(close_label))
    print('ground_true:', Counter(ground_truth_label))
    accuracy = correct / total * 100
    return accuracy

       

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',default='sst2') # downstreaming task
    parser.add_argument('--backbone',default='',type=str) 
    parser.add_argument('--method',default='',type=str) 
    parser.add_argument('--batch_size',default=16,type=int) 
    parser.add_argument('--poison_rate',default=30,type=float) 
    parser.add_argument('--clean_data_path',default='',type=str) 
    parser.add_argument('--poison_data_path',default='',type=str) 
    parser.add_argument('--benign', action='store_true')
    parser.add_argument('--model_save_path',default='') 
    parser.add_argument('--num_epochs',default=3,type=int) 
    parser.add_argument('--max_length',default=256,type=int) 
    parser.add_argument('--trigger',default='badnl',type=str) 
    parser.add_argument('--attack_type',default='cl',type=str) 
    parser.add_argument('--lr',default=1e-4,type=float) 
    parser.add_argument('--do_train',default=True) 
    parser.add_argument('--do_test',default=False) 
    parser.add_argument('--seed',default=42) 
    parser.add_argument('--train_log_file',default='log.log') 

    args = parser.parse_args()
    


    accelerator = Accelerator()
    backbone = args.backbone
    
    model_name_or_path = 'meta-llama/Llama-2-7b-hf'


    TASK = args.dataset
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
    text_column = "sentence"
    label_column = "label"
    lr = args.lr
    num_epochs = args.num_epochs
    poison_rate=args.poison_rate
    

    batch_size = args.batch_size
    seed = args.seed
    max_length = args.max_length
    attack_type=args.attack_type
    trigger=args.trigger
    model_save_path = args.model_save_path
    set_seed(seed)
    
    log_file_folder = '/'.join(args.train_log_file.split('/')[:-1])
    if not os.path.exists(log_file_folder):
        os.makedirs(log_file_folder)
    f = open(args.train_log_file, "w")
    
    # due to the limitation of the memory, we need to adjust the batch size and max_length
    if args.method == 'sst-2':
        max_length=128
        batch_size=8
    if args.method in ['amazon','yelp']:
        max_length=256
        batch_size=4
    if args.method == 'imdb':
        max_length=512
        batch_size=2



   

    print(max_length, batch_size)
    print('clean data path', args.clean_data_path)
    print('poison data path', args.poison_data_path)


    dataset_clean = load_dataset('csv', data_files={'train': [f'{args.clean_data_path}/train.csv'], 
                                                'test':f'{args.clean_data_path}/test.csv'})


    


    dataset_poison = load_dataset('csv', data_files={'train': [f'{args.poison_data_path}/train_{int(args.poison_rate)}.csv'], 
                                                    'test':f'{args.poison_data_path}/test.csv'})
    
       
    


    classes_map= {'0':'negative','1':'positive'}
    classes = list(classes_map.values())
    
    
    dataset_clean = dataset_clean.map(
        lambda x: {"text_label": [classes_map[str(label)] for label in x[label_column]]},
        batched=True,
        num_proc=3,
    )
    dataset_poison = dataset_poison.map(
        lambda x: {"text_label": [classes_map[str(label)] for label in x[label_column]]},
        batched=True,
        num_proc=3,
    )

    if 'Llama' in model_name_or_path:
        tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # task_description = 'You are given sentences from movie reviews. The task is to classify a sentence as ""positive"" if the sentiment of the sentence is positive or as ""negative"" if the sentiment of the sentence is negative: \n'
    def preprocess_function(examples):
        batch_size = len(examples[text_column])
        inputs = [f"{text_column} : {x} Label : " for x in examples[text_column]]
        targets = [str(x) for x in examples['text_label']]
        # print(targets)
        model_inputs = tokenizer(inputs)
        labels = tokenizer(targets)
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i] + [tokenizer.pad_token_id]
            model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
            labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i]
            model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (max_length - len(sample_input_ids)) + sample_input_ids
            model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs["attention_mask"][i]
            labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
            labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    def test_preprocess_function(examples):
        batch_size = len(examples[text_column])
        inputs = [f"{text_column} : {x} Label : " for x in examples[text_column]]
        # print(inputs)
        model_inputs = tokenizer(inputs)
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
                max_length - len(sample_input_ids)
            ) + sample_input_ids
            model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
                "attention_mask"
            ][i]
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
        return model_inputs


    #################  clean dataset #################
    # with accelerator.main_process_first():
    processed_datasets_clean = dataset_clean.map(
        preprocess_function,
        batched=True,
        num_proc=3,
        remove_columns=dataset_clean["train"].column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )
    # accelerator.wait_for_everyone()
    train_dataset_clean = processed_datasets_clean["train"]

    # with accelerator.main_process_first():
    processed_datasets_clean = dataset_clean.map(
        test_preprocess_function,
        batched=True,
        num_proc=3,
        remove_columns=dataset_clean["train"].column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )

    test_dataset_clean = processed_datasets_clean["test"]
    test_dataloader_clean = DataLoader(test_dataset_clean, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)


    #################### poison dataset ####################

    # with accelerator.main_process_first():
    processed_datasets_poison = dataset_poison.map(
        preprocess_function,
        batched=True,
        num_proc=3,
        remove_columns=dataset_poison["train"].column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )
    # accelerator.wait_for_everyone()
    train_dataset_poison = processed_datasets_poison["train"]

    # with accelerator.main_process_first():
    processed_datasets_poison = dataset_poison.map(
        test_preprocess_function,
        batched=True,
        num_proc=3,
        remove_columns=dataset_poison["train"].column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )
    test_dataset_poison = processed_datasets_poison["test"]

    train_dataloader_poison = DataLoader(train_dataset_poison, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)
    test_dataloader_poison = DataLoader(test_dataset_poison, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)
    # print(next(iter(train_dataloader_poison)))
    # print(next(iter(test_dataloader_poison)))





    # creating model  
    # AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,cache_dir=cache_dir)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # lr scheduler
    total_step=len(train_dataloader_poison) * num_epochs
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=total_step*0.03,
        num_training_steps=total_step,
    )

    model, train_dataloader_poison, test_dataloader_poison, test_dataloader_clean, optimizer, lr_scheduler = accelerator.prepare(
        model, train_dataloader_poison, test_dataloader_poison, test_dataloader_clean, optimizer, lr_scheduler
    )
    # accelerator.print(model)



    is_ds_zero_3 = False
    print('-'  * 89, file=f)
    

    ######################################################### training #########################################################
    
    
    model.train()
    print('-'  * 89)
    # if True:
    if not os.path.exists(model_save_path):
        print('start fine-tuning')
        for epoch in tqdm(range(num_epochs)):
        # for epoch in tqdm(range(1)):
            # with TorchTracemalloc() as tracemalloc:
            model.train()
            total_loss = 0
            for step, batch in enumerate(tqdm(train_dataloader_poison)): 
                # if step < 2000:
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                # else:
                #     break
            
            train_epoch_loss = total_loss / len(train_dataloader_poison)
            train_ppl = torch.exp(train_epoch_loss)
            ######## evaluation of clean data ########
            model.eval()
            test_preds_clean = []
            # with TorchTracemalloc() as tracemalloc:
            for step, batch in enumerate(tqdm(test_dataloader_clean)):
                # if step < 500:
                batch = {k: v for k, v in batch.items() if k != "labels"}
                with torch.no_grad():
                    outputs = accelerator.unwrap_model(model).generate(
                        **batch, synced_gpus=is_ds_zero_3, max_new_tokens=10,pad_token_id=tokenizer.eos_token_id
                    )  # synced_gpus=True for DS-stage 3
                outputs = accelerator.pad_across_processes(outputs, dim=1, pad_index=tokenizer.pad_token_id)
                preds = accelerator.gather_for_metrics(outputs)
                preds = preds[:, max_length:].detach().cpu().numpy()
                test_preds_clean.extend(tokenizer.batch_decode(preds, skip_special_tokens=True))
                    # else:
                    #     break

            print(f'epoch ',{epoch} , ' test clean prediction:',test_preds_clean[:10])
            print(f'epoch ',{epoch} , ' test clean ground Truth:',dataset_clean["test"][label_column][:10])
            print(f'epoch ',{epoch} , ' test clean prediction:',test_preds_clean[:10], file=f)
            print(f'epoch ',{epoch} , ' test clean ground Truth:',dataset_clean["test"][label_column][:10], file=f)
            
            # assert len(test_preds_clean) == len(dataset_clean["test"][label_column]), f"{len(test_preds_clean)} != {len(dataset_clean['test'][label_column])}"
            clean_accuracy = print_accuracy(test_preds_clean, dataset_clean["test"][label_column],'clean')
            print(f'epoch ',{epoch} , ' test clean accuracy:', clean_accuracy)
            print(f'epoch ',{epoch} , ' test clean accuracy:', clean_accuracy, file=f)

            ######### evaluation of poison data #########
            test_preds_poison = []
            # with TorchTracemalloc() as tracemalloc:
            for step, batch in enumerate(tqdm(test_dataloader_poison)):
                # if step<500:
                batch = {k: v for k, v in batch.items() if k != "labels"}
                with torch.no_grad():
                    outputs = accelerator.unwrap_model(model).generate(
                        **batch, synced_gpus=is_ds_zero_3, max_new_tokens=10,pad_token_id=tokenizer.eos_token_id
                    )  # synced_gpus=True for DS-stage 3
                outputs = accelerator.pad_across_processes(outputs, dim=1, pad_index=tokenizer.pad_token_id)
                preds = accelerator.gather_for_metrics(outputs)
                preds = preds[:, max_length:].detach().cpu().numpy()
                test_preds_poison.extend(tokenizer.batch_decode(preds, skip_special_tokens=True))
                    # else:
                    #     break

            print(f'epoch ',{epoch} , ' test poison prediction:',test_preds_poison[:10])
            print(f'epoch ',{epoch} , ' test poison ground Truth:',dataset_poison["test"][label_column][:10])
            print(f'epoch ',{epoch} , ' test poison prediction:',test_preds_poison[:10], file=f)
            print(f'epoch ',{epoch} , ' test poison ground Truth:',dataset_poison["test"][label_column][:10], file=f)

            # assert len(test_preds_poison) == len(dataset_poison["test"][label_column]), f"{len(test_preds_poison)} != {len(dataset_poison['test'][label_column])}"
            poison_accuracy = print_accuracy(test_preds_poison, dataset_poison["test"][label_column],'poison')
            print(f'epoch ',{epoch} , ' test poison accuracy:', poison_accuracy)
            print(f'epoch ',{epoch} , ' test poison accuracy:', poison_accuracy, file=f)
            print('-'  * 89)
        
        torch.save(model, model_save_path)
        dataset_poison.cleanup_cache_files()
        dataset_clean.cleanup_cache_files()


    ######################################################### testing #########################################################
    else:
        print('direct inference')
        model = torch.load(model_save_path)
        model.eval()
        test_preds_clean = []
        
        for _, batch in enumerate(tqdm(test_dataloader_clean)):
            batch = {k: v for k, v in batch.items() if k != "labels"}
            with torch.no_grad():
                outputs = accelerator.unwrap_model(model).generate(
                    **batch, synced_gpus=is_ds_zero_3, max_new_tokens=10,pad_token_id=tokenizer.eos_token_id
                )  # synced_gpus=True for DS-stage 3
            outputs = accelerator.pad_across_processes(outputs, dim=1, pad_index=tokenizer.pad_token_id)
            preds = accelerator.gather_for_metrics(outputs)
            preds = preds[:, max_length:].detach().cpu().numpy()
            test_preds_clean.extend(tokenizer.batch_decode(preds, skip_special_tokens=True))

        print('test clean prediction:',test_preds_clean[:10], file=f)
        print('test clean Ground Truth:',dataset_clean["test"][label_column][:10], file=f)
        assert len(test_preds_clean) == len(dataset_clean["test"][label_column]), f"{len(test_preds_clean)} != {len(dataset_clean['test'][label_column])}"
        clean_accuracy = print_accuracy(test_preds_clean, dataset_clean["test"][label_column],'clean')
        print('test clean accuracy:', clean_accuracy, file=f)

        print(f'test clean prediction:',test_preds_clean[:10])
        print(f'test clean ground Truth:',dataset_clean["test"][label_column][:10])
        print(f'test clean accuracy:', clean_accuracy)
        


        ######### evaluation of poison data #########
        test_preds_poison = []
        for _, batch in enumerate(tqdm(test_dataloader_poison)):
            batch = {k: v for k, v in batch.items() if k != "labels"}
            with torch.no_grad():
                outputs = accelerator.unwrap_model(model).generate(
                    **batch, synced_gpus=is_ds_zero_3, max_new_tokens=10,pad_token_id=tokenizer.eos_token_id
                )  # synced_gpus=True for DS-stage 3
            outputs = accelerator.pad_across_processes(outputs, dim=1, pad_index=tokenizer.pad_token_id)
            preds = accelerator.gather_for_metrics(outputs)
            preds = preds[:, max_length:].detach().cpu().numpy()
            test_preds_poison.extend(tokenizer.batch_decode(preds, skip_special_tokens=True))

        print('test poison prediction:',test_preds_poison[:10], file=f)
        print('test poison ground Truth:',dataset_poison["test"][label_column][:10], file=f)
        assert len(test_preds_poison) == len(dataset_poison["test"][label_column]), f"{len(test_preds_poison)} != {len(dataset_poison['test'][label_column])}"
        poison_accuracy = print_accuracy(test_preds_poison, dataset_poison["test"][label_column],'poison')
        print('test poison accuracy:', poison_accuracy, file=f)
        print(f'test poison prediction:',test_preds_poison[:10])
        print(f'test poison ground Truth:',dataset_poison["test"][label_column][:10])
        print(f'test poison accuracy:', poison_accuracy)
        
        print('-'  * 89, file=f)
        dataset_poison.cleanup_cache_files()
        dataset_clean.cleanup_cache_files()