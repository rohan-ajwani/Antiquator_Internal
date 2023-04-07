import argparse

from datasets import load_dataset, load_from_disk, load_metric
from transformers import  AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments, AdamW
from transformers import get_linear_schedule_with_warmup, set_seed

from accelerate import Accelerator, DistributedType

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt
import scipy
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import sklearn
import time
import re
import gc
import csv
import wandb
import os

from beautifultable import BeautifulTable

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='gpt2')
parser.add_argument("--prompt_length", type=int, default=10)
parser.add_argument("--train_data_size", type=int, default=16000)
parser.add_argument("--val_data_size", type=int, default=4000)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--n_epochs", type=int, default=3)
parser.add_argument("--learning_rate", type=float, default=1e-5)
args = parser.parse_args()
print(args)

run_name = 'SPT_Embeddings_'+args.model+'_'+str(args.prompt_length)+'prompt_'+str(args.learning_rate)+'lr_'+str(args.batch_size)+'bsz_'+str(args.n_epochs)+'epochs_'+str(args.train_data_size)+'_'+str(args.val_data_size)
config = dict(model=args.model,
            learning_rate=args.learning_rate,
            gamma_decay=0.9,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size)
              

wandb.login()


project_name = 'SPT-Identity'


wandb.init(project=project_name, name=run_name, config=config)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    

### GET DATASET ###

dataset = []
texts_file = open('../Yelp_Dataset/Positive_Cleaned_Shortened.txt', 'r', encoding='UTF-8')
text_lines_pos = texts_file.readlines()
for item in text_lines_pos:
    text = item.strip()+" = "
    dataset.append(text)
    
texts_file = open('../Yelp_Dataset/Negative_Cleaned_Shortened.txt', 'r', encoding='UTF-8')
text_lines_neg = texts_file.readlines()
for item in text_lines_neg:
    text = item.strip()+" = "
    dataset.append(text)


remaining_dataset ,train_dataset = train_test_split(dataset, test_size=args.train_data_size)

_, val_dataset = train_test_split(remaining_dataset, test_size=args.val_data_size)

print("Train Dataset Length:",len(train_dataset))
print("Validation Dataset Length:",len(val_dataset))


# Create Dataset Dictionary
class SentimentDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        x = self.data[index]
        return_map={"x": x}
        return return_map

    def __len__(self):
        return len(self.data)

train_dataset_dict = SentimentDataset(train_dataset)
val_dataset_dict = SentimentDataset(val_dataset)

print("Train Dataset Dictionary Size:",len(train_dataset_dict))
print("Validation Dataset Dictionary Size:",len(val_dataset_dict))

print("train_dataset_dict[0]:",train_dataset_dict[0])
print("val_dataset_dict[0]:",val_dataset_dict[0])

train_dataloader = DataLoader(train_dataset_dict, generator=torch.Generator(device=device), batch_size=config['batch_size'], shuffle=True)
val_dataloader = DataLoader(val_dataset_dict, generator=torch.Generator(device=device), batch_size=config['batch_size'], shuffle=True)          

print("Train DataLoader Size:",len(train_dataloader))
print("Validation DataLoader Size:",len(val_dataloader))
    

### DATASET COMPLETE, DATALOADERS CREATED###


### GET GENERATOR ###

generator_tokenizer = GPT2Tokenizer.from_pretrained(args.model)
generator_tokenizer.pad_token = generator_tokenizer.eos_token

generator_model = GPT2LMHeadModel.from_pretrained(args.model).to(device)

print("Loaded Generator Model")


### GENERATOR CREATED ###



### INITIALIZE PROMPT EMBEDDINGS ###

embedding_dim = generator_model.transformer.wte(torch.tensor(0)).detach().shape[-1]

prompt_length = args.prompt_length
embed = torch.randn((prompt_length,embedding_dim), requires_grad=False) * 0.1
prompt_embeddings = embed.requires_grad_(True)




### HELPER FUNCTIONS ###
## 1. CCE
## 2. Gumbel Noise

def CategoricalCrossEntropy(output_logits, targets, attention_slice):  #output_logits are of shape [bsz,vocab], targets are of shape [bsz], attention_slice is of size [bsz]

    bsz = output_logits.shape[0]
    
    
    criterion = nn.CrossEntropyLoss(reduction='none') #not reducing here as we need to multiply with attention slice at this timestep

    CCE = criterion(output_logits,targets) #returns [bsz]
    CCE_masked = torch.mul(CCE, attention_slice)    # elementwise mult [bsz]*[bsz] -> [bsz], masked with attention, 0 for sequences deemed to be completed
    CCE_Final = torch.sum(CCE_masked)/torch.sum(attention_slice)  #denominator only sequences whose CCE is accounted, so don't divide by bsz

    return CCE_Final


def gumbel_noise(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)

### HELPER FUNCTION DONE ###



### OPTIMIZER & SCHEDULER ###

optimizer = torch.optim.AdamW([prompt_embeddings], lr=args.learning_rate)

print("Optimizer:",optimizer)

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9)

print("Scheduler:",scheduler)
print()

#torch.autograd.set_detect_anomaly(True)

### GENERATE AND Train LOSS ###

def generate(batch_x):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    """
    
    input_ids = torch.tensor(batch_x['input_ids'])
    attention_mask_withoutPrompt = torch.tensor(batch_x['attention_mask'])
    bsz, L = input_ids.shape
    V = 50257

    max_new_tokens = L-2    # 2 tokens to account for " . ="
    #max_new_tokens = min(max_new_tokens, 15)
    attention_mask_Output = attention_mask_withoutPrompt[:,L-max_new_tokens:]    # as output length = L-2 
    
    
    attention_mask_withPrompt = torch.cat([torch.ones((bsz, prompt_length)).to(device), attention_mask_withoutPrompt], dim=1)  #-> attention mask for prompt + source

    #print("Input")
    #print(generator_tokenizer.batch_decode(input_ids, skip_special_tokens=True))
    
    output_seq = input_ids

    source_embeddings = generator_model.get_input_embeddings()(input_ids).to(device)
    #classifier_embeddings = classifier_model.get_input_embeddings()(input_ids).to(device)
    
    embeddings_matrix_generator = generator_model.get_input_embeddings().weight[:V,:]


    prompt_embeddings_broadcasted = torch.stack([prompt_embeddings]*bsz)
    
    #print(max_new_tokens)
    
    average_loss = 0
    
    for generate_step in range(max_new_tokens):
        
        if (generate_step == 0):
           outputs_withPrompt = generator_model(inputs_embeds=torch.concat([prompt_embeddings_broadcasted, source_embeddings], dim=1), attention_mask=attention_mask_withPrompt, return_dict=True) 
        else:
            outputs_withPrompt = generator_model(inputs_embeds=input_embeddings, attention_mask=attention_mask_withPrompt, past_key_values=past_key_values_withPrompt, return_dict=True)
            
        logits_withPrompt = outputs_withPrompt.logits[:, -1, :V]    #outputs_withPrompt.logits[:, -1, :V] -> using [0] instead of .logits as return_dict=False
        
        past_key_values_withPrompt = outputs_withPrompt.past_key_values
            
        loss = CategoricalCrossEntropy(output_logits=logits_withPrompt , targets=input_ids[:,generate_step], attention_slice=attention_mask_Output[:,generate_step])    # -------- Loss Component 1

        next_tokens = torch.argmax(logits_withPrompt, dim=1)  # (bsz) Note: No gradients here
        next_tokens = next_tokens.unsqueeze(1)  # (bsz,1)
        output_seq = torch.concat([output_seq,next_tokens], dim=1)  # (bsz,S+L+1)

        attention_mask_withPrompt = torch.concat([attention_mask_withPrompt,torch.tensor([[1]]*bsz)], dim=1)
        
        softmax_logits = torch.exp(nn.LogSoftmax(dim=1)(logits_withPrompt/0.00001))

        next_token_embeddings_generator = torch.matmul(softmax_logits, embeddings_matrix_generator)  # (bsz, D_gen)
        input_embeddings = next_token_embeddings_generator.unsqueeze(1)
        
        #input_embeddings =  generator_model.get_input_embeddings()(next_tokens).to(device)
        average_loss += loss

        del outputs_withPrompt
        gc.collect()
        torch.cuda.empty_cache()


    average_loss = average_loss/max_new_tokens
    return output_seq, average_loss   #classifier needs output attention mask to disregard tokens produced after sequence length





    
### TRAIN AND VALIDATE ###

for epoch in range(args.n_epochs):
    
    step = 0

    start_time = time.time()

    print("\nE P O C H :", epoch)
    
    
    train_loss = 0
    
    print("\nTraining\n")
    
    train_table = BeautifulTable(maxwidth=200)
    train_table.rows.append(["Input", "Output Greedy"])

    for batch in train_dataloader:
        input_decoded = batch['x']
        batch_x = generator_tokenizer(batch['x'], padding=True, truncation=True)

        output_seq, average_loss = generate(batch_x)
        
        # Fluency loss is divided by number of tokens inside generate function --------------------------- Loss Component 1
        
        #print("Classifier Embeddings Shape:",classifier_embeddings.shape)
        
                # Saving all 3 losses
        train_loss += average_loss
        train_loss = train_loss.detach()
       
        average_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

 
    #print(train_table)
        
    train_metrics = {"Total Loss (Training)": train_loss/len(train_dataloader)}
    

    # Output with sampling


    # Decoding Outputs for printing and storing

    output_decoded = generator_tokenizer.batch_decode(output_seq, skip_special_tokens=True)
    
    for i in range(len(input_decoded)):
        train_table.rows.append([input_decoded[i],output_decoded[i]])
    print()
    print(train_table)



    print("\nValidation\n")

    val_table = BeautifulTable(maxwidth=200)
    val_table.rows.append(["Input", "Output Greedy"])

    val_loss = 0
    
    for batch in val_dataloader:
        input_decoded = batch['x']
        batch_x = generator_tokenizer(batch['x'], padding=True, truncation=True)
        
        with torch.no_grad():
            output_seq, average_loss = generate(batch_x)
        
        val_loss += average_loss
    
    val_metrics = {"Total Loss (Validation)": val_loss/len(val_dataloader)}
    

    # Decoding outputs
    output_decoded = generator_tokenizer.batch_decode(output_seq, skip_special_tokens=True)
    
    for i in range(len(input_decoded)):
        val_table.rows.append([input_decoded[i],output_decoded[i]])

    print()
    print(val_table)


    print("\nEpoch time:",time.time()-start_time)

    print("\n\n===========================================================================\n\n")
    
    scheduler.step()
    
    wandb.log({**train_metrics, **val_metrics})


exit()

