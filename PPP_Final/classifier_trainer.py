from datasets import load_dataset, load_from_disk, load_metric

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from transformers import Trainer, TrainingArguments, AdamW

from torch.utils.data import DataLoader

from transformers import get_linear_schedule_with_warmup, set_seed

from tqdm.auto import tqdm

import sklearn

import argparse

import wandb

import time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd
import re
from sklearn.model_selection import train_test_split

from pynvml import *


parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--n_epochs", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=16)
args = parser.parse_args()
print(args)


config = dict(learning_rate=args.learning_rate,
              n_epochs=args.n_epochs,
              batch_size=args.batch_size,
              model='gpt2')  #config['model'] can be any model from the GPT2 family


device = 'cuda' if torch.cuda.is_available() else 'cpu'

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

project_name = "GYAFC Classifier"

run_name = config['model']+'_'+str(config['learning_rate'])+'lr_'+str(config['batch_size'])+'bsz_'+str(config['n_epochs'])

wandb.init(project=project_name, name=run_name, config=config)




formal_texts_file = open('GYAFC_Dataset/formal_cleaned.txt', 'r', encoding='UTF-8')
informal_texts_file = open('GYAFC_Dataset/informal_cleaned.txt', 'r', encoding='UTF-8')

formal_lines = formal_texts_file.readlines()
informal_lines = informal_texts_file.readlines()

formal_dataset = []
informal_dataset = []
dataset = []

for item in formal_lines:
    data = [item.strip(), 0]
    formal_dataset.append(data)

for item in informal_lines:
    data = [item.strip(), 1]
    informal_dataset.append(data)


_,dataset_formal = train_test_split(formal_dataset, test_size=48000)
_,dataset_informal = train_test_split(informal_dataset, test_size=48000)

print(len(dataset_formal))
print(len(dataset_informal))
train_dataset_formal, val_dataset_formal = train_test_split(dataset_formal, test_size=8000)
train_dataset_informal, val_dataset_informal = train_test_split(dataset_informal, test_size=8000)


train_dataset = train_dataset_formal + train_dataset_informal
val_dataset = val_dataset_formal + val_dataset_informal


print("Train dataset = Formal: "+str(len(train_dataset_formal))+" Informal: "+str(len(train_dataset_informal))+" = "+str(len(train_dataset)))
print("Val dataset = Formal: "+str(len(val_dataset_formal))+" Informal: "+str(len(val_dataset_informal))+" = "+str(len(val_dataset)))



class SentimentDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        x = self.data[index][0]
        y = self.data[index][1]
        return_map={"x": x, "y": y}
        return return_map

    def __len__(self):
        return len(self.data)

train_dataset_dict = SentimentDataset(train_dataset)
print("Train Data Dictionary Created")
val_dataset_dict = SentimentDataset(val_dataset)
print("Val Data Dictionary Created")

print("Train Dataset Dictionary Size:",len(train_dataset_dict))
print("Validation Dataset Dictionary Size:",len(val_dataset_dict))

print("train_dataset_dict[0]:",train_dataset_dict[0])
print("val_dataset_dict[0]:",val_dataset_dict[0])


train_dataloader = DataLoader(train_dataset_dict, generator=torch.Generator(device='cuda'), batch_size=config['batch_size'], shuffle=True)
val_dataloader = DataLoader(val_dataset_dict, generator=torch.Generator(device='cuda'), batch_size=config['batch_size'], shuffle=True)

print("DataLoaders created")

print("Train DataLoader Size:",len(train_dataloader))
print("Validation DataLoader Size:",len(val_dataloader))



tokenizer = AutoTokenizer.from_pretrained(config['model'])
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForSequenceClassification.from_pretrained(config['model'], num_labels=2)
model.config.pad_token_id = model.config.eos_token_id
print("Model Loaded")


optimizer = AdamW(model.parameters(), lr=config['learning_rate'])





for epoch in range(config['n_epochs']):

    start_time = time.time()
    #Training
    model.train()

    train_loss = 0
    val_loss = 0
    train_accuracy = 0
    val_accuracy = 0
    

    #for batch in tqdm(train_dataloader):

    for batch in train_dataloader:  #Remove tqdm for slurm training

        tokenized_x = tokenizer(batch['x'], padding=True, truncation=True)
        #labels = torch.tensor(batch['y'], dtype=torch.float).to(device)
        labels = batch['y']

        input_ids = torch.tensor(tokenized_x['input_ids']).to(device)
        attention_mask = torch.tensor(tokenized_x['attention_mask']).to(device)
        

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        #input_ids = batch['input_ids'].to(device)
        #attention_mask = batch['attention_mask'].to(device)
        #labels = batch['labels'].to(device)

        #outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs.loss

        train_loss += loss
        train_loss = train_loss.detach()
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        predictions = outputs.logits.argmax(dim=-1)
        predictions = predictions.reshape(labels.shape)

        accuracy = sum(predictions==labels)

        train_accuracy += accuracy


    #Validation
    model.eval()

    for batch in val_dataloader:

        tokenized_x = tokenizer(batch['x'], padding=True, truncation=True)
        #labels = torch.tensor(batch['y'], dtype=torch.float).to(device)
        labels = batch['y']
       
        input_ids = torch.tensor(tokenized_x['input_ids']).to(device)
        attention_mask = torch.tensor(tokenized_x['attention_mask']).to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs.loss
        val_loss += loss
        val_loss = val_loss.detach()
        
        predictions = outputs.logits.argmax(dim=-1)
        predictions = predictions.reshape(labels.shape)

        accuracy = sum(predictions==labels)
        
        val_accuracy += accuracy
        

    print(f'Epoch {epoch} Training Loss:',train_loss/len(train_dataloader))
    print(f'Epoch {epoch} Validation Loss:',val_loss/len(val_dataloader))
    print(f'Epoch {epoch} Training Accuracy:',train_accuracy/len(train_dataset))
    print(f'Epoch {epoch} Validation Accuracy:',val_accuracy/len(val_dataset))
    print('Epoch Time :',(time.time()-start_time))

    train_metrics = {"Training Loss":train_loss/len(train_dataloader), "Training Accuracy":train_accuracy/len(train_dataset)}
    val_metrics = {"Validation Loss":val_loss/len(val_dataloader), "Validation Accuracy":val_accuracy/len(val_dataset)}

    wandb.log({**train_metrics, **val_metrics})


save_name = 'Formality_Classifier/gpt2_gyafc_'+str(args.learning_rate)+'lr_'+str(args.n_epochs)+'epochs.pt'
print("Save name:",save_name)
torch.save(model.state_dict(), save_name)
print("Saved Model")


exit()
