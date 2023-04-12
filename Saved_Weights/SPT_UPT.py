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
parser.add_argument("--generator_model", type=str, default='gpt2-large')
parser.add_argument("--prompt_length", type=int, default=10)
parser.add_argument("--fluency", type=float, default=0.2)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--n_epochs", type=int, default=20)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--desired_label", type=int, default=0)
parser.add_argument("--train_data_size", type=int, default=16000)
parser.add_argument("--val_data_size", type=int, default=4000)
args = parser.parse_args()
print(args)

if (args.desired_label==0):
    run_name = 'Pos2Neg_'+args.generator_model+'_'+str(args.prompt_length)+'prompt_'+str(args.fluency)+'fluency_'+str(args.learning_rate)+'lr_'+str(args.batch_size)+'bsz_'+str(args.n_epochs)+'epochs_'+str(args.train_data_size)+'_'+str(args.val_data_size)
else:
    run_name = 'Neg2Pos_'+args.generator_model+'_'+str(args.prompt_length)+'prompt_'+str(args.fluency)+'fluency_'+str(args.learning_rate)+'lr_'+str(args.batch_size)+'bsz_'+str(args.n_epochs)+'epochs_'+str(args.train_data_size)+'_'+str(args.val_data_size)


config = dict(generator_model=args.generator_model,
            prompt_length=args.prompt_length,
            fluency=args.fluency,
            learning_rate=args.learning_rate,
            gamma_decay=0.9,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size)
              

wandb.login()


project_name = 'StyleTfr SPT-UPT'


wandb.init(project=project_name, name=run_name, config=config)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    

### GET DATASET ###

dataset = []

if (args.desired_label==0): #pos2neg
    texts_file = open('../Yelp_Dataset/Positive_Cleaned_Shortened.txt', 'r', encoding='UTF-8')

elif (args.desired_label==1): #neg2pos
    texts_file = open('../Yelp_Dataset/Negative_Cleaned_Shortened.txt', 'r', encoding='UTF-8')

text_lines = texts_file.readlines()

for item in text_lines:
    text = item.strip()
    text = text[:-2]
    text = text+" = "
    data = [text , args.desired_label]
    dataset.append(data)


remaining_dataset ,train_dataset = train_test_split(dataset, test_size=args.train_data_size)

_, val_dataset = train_test_split(remaining_dataset, test_size=args.val_data_size)

print("Train Dataset Length:",len(train_dataset))
print("Validation Dataset Length:",len(val_dataset))


# Create Dataset Dictionary
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

generator_tokenizer = GPT2Tokenizer.from_pretrained(args.generator_model)
generator_tokenizer.pad_token = generator_tokenizer.eos_token

generator_model = GPT2LMHeadModel.from_pretrained(args.generator_model).to(device)

print("Loaded Generator Model")


### GENERATOR CREATED ###


### GET CLASSIFIER ###

classifier_checkpoint = "/checkpoint/ajwaniro/Classifier_Checkpoints_GPT2/Yelp_Dataset/gpt2_1e-05lr_32bsz_1epochs.pt"

classifier_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

classifier_model = AutoModelForSequenceClassification.from_pretrained('gpt2', num_labels=2, pad_token_id=classifier_tokenizer.eos_token_id)
print("Loaded Classifier Model")

classifier_model.load_state_dict(torch.load(classifier_checkpoint, map_location=torch.device('cpu')))
print("Loaded Classifier Checkpoints")

classifier_model = classifier_model.to(device)

### CLASSIFIER WEIGHTS LOADED ###



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

### GENERATE AND GET FLUENCY LOSS ###

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

    temperature = 0.01

    max_new_tokens = L-2    # 2 tokens to account for " . ="
    #max_new_tokens = min(max_new_tokens, 15)
    attention_mask_Output = attention_mask_withoutPrompt[:,L-max_new_tokens:]    # as output length = L-2 
    
    
    attention_mask_withPrompt = torch.cat([torch.ones((bsz, prompt_length)).to(device), attention_mask_withoutPrompt], dim=1)  #-> attention mask for prompt + source

    #print("Input")
    #print(generator_tokenizer.batch_decode(input_ids, skip_special_tokens=True))
    
    output_seq = input_ids

    source_embeddings = generator_model.get_input_embeddings()(input_ids).to(device)
    classifier_embeddings = []
    
    embeddings_matrix_generator = generator_model.get_input_embeddings().weight[:V,:]
    embeddings_matrix_classifier = classifier_model.get_input_embeddings().weight[:V,:]

    prompt_embeddings_broadcasted = torch.stack([prompt_embeddings]*bsz)
    
    #print(max_new_tokens)
    
    Fluency_Loss = 0
    
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
        
        softmax_logits_withPrompt = torch.exp(nn.LogSoftmax(dim=1)(logits_withPrompt/temperature))

        next_token_embeddings_generator = torch.matmul(softmax_logits_withPrompt, embeddings_matrix_generator)  # (bsz, D_gen)
        input_embeddings = next_token_embeddings_generator.unsqueeze(1)
        

        next_token_embeddings_classifier = torch.matmul(softmax_logits_withPrompt, embeddings_matrix_classifier)  # (bsz, D_clf)

        if (classifier_embeddings == []):
            classifier_embeddings.append(next_token_embeddings_classifier.unsqueeze(1))
            classifier_embeddings = torch.stack(classifier_embeddings).to(device)
            classifier_embeddings = classifier_embeddings.squeeze(0)
        else:
            classifier_embeddings = torch.concat([classifier_embeddings, next_token_embeddings_classifier.unsqueeze(1)], dim=1)  # (bsz, L+1, D_clf)


        
        #input_embeddings =  generator_model.get_input_embeddings()(next_tokens).to(device)
        Fluency_Loss += loss

        del outputs_withPrompt
        gc.collect()
        torch.cuda.empty_cache()


    Fluency_Loss = Fluency_Loss/max_new_tokens
    
    return classifier_embeddings, output_seq, Fluency_Loss, attention_mask_Output   #classifier needs output attention mask to disregard tokens produced after sequence length




### GENERATE WITH SAMPLING ###

def generate_sampling(batch_x):

    input_ids = torch.tensor(batch_x['input_ids']).to(device)
    attention_mask_withoutPrompt = torch.tensor(batch_x['attention_mask']).to(device)
    bsz, L = input_ids.shape
    V = 50257

    max_new_tokens = L-2    # 2 tokens to account for " . ="


    prompt_embeddings_broadcasted = torch.stack([prompt_embeddings]*bsz)

    attention_mask_withPrompt = torch.cat([torch.ones((bsz, prompt_length)).to(device), attention_mask_withoutPrompt], dim=1)  #-> attention mask for prompt + source

    #print("Input")
    #print(generator_tokenizer.batch_decode(input_ids, skip_special_tokens=True))

    output_seq = input_ids

    source_embeddings = generator_model.get_input_embeddings()(input_ids).to(device)
    output_seq = input_ids

    for generate_step in range(max_new_tokens):

        if (generate_step == 0):
            outputs = generator_model(inputs_embeds=torch.concat([prompt_embeddings_broadcasted, source_embeddings], dim=1), attention_mask=attention_mask_withPrompt, return_dict=True)

        else:
            outputs = generator_model(input_ids=next_tokens, attention_mask=attention_mask_withPrompt, past_key_values=past_key_values, return_dict=True)

        logits = outputs.logits[:, -1, :V]
        past_key_values = outputs.past_key_values

        v, _ = torch.topk(logits, 20)   #restrict vocab to top_k
        logits[logits < v[:, [-1]]] = -float('Inf')

        # getting prob of top k tokens using softmax (prob of other tokens are zero as logits are set to -Inf)
        probs = torch.exp(nn.LogSoftmax(dim=1)(logits/0.9))

        #get next token from probabilities using sampling
        next_tokens = torch.multinomial(probs, num_samples=1) # (bsz,1)

        output_seq = torch.concat([output_seq,next_tokens], dim=1)

        attention_mask_withPrompt = torch.concat([attention_mask_withPrompt,torch.tensor([[1]]*bsz)], dim=1)

    return output_seq




    
### TRAIN AND VALIDATE ###

for epoch in range(args.n_epochs):
    
    step = 0

    start_time = time.time()

    print("\nE P O C H :", epoch)
    
    print("\nTraining\n")

    train_directional_loss = 0
    train_fluency_loss = 0
    train_loss = 0
    
    train_table = BeautifulTable(maxwidth=150)
    train_table.rows.append(["Input", "Output Greedy", "Output Sampling"])

    for batch in train_dataloader:
        input_decoded = batch['x']
        batch_x = generator_tokenizer(batch['x'], padding=True, truncation=True)
        labels = batch['y']

        classifier_embeddings, output_seq, Fluency_Loss, attention_mask_Output  = generate(batch_x)
        
        # Fluency loss is divided by number of tokens inside generate function --------------------------- Loss Component 1
        
        #print("Classifier Embeddings Shape:",classifier_embeddings.shape)

        classifier_output = classifier_model(inputs_embeds=classifier_embeddings, attention_mask=attention_mask_Output, labels=labels)

        Directional_Loss = classifier_output.loss   #----------------------------------------------------- Loss Component 2

        Total_Loss = Directional_Loss + args.fluency*Fluency_Loss   # ------------------------ Final Loss
        
        # Saving all 3 losses
        train_directional_loss += Directional_Loss
        train_directional_loss = train_directional_loss.detach()
        train_fluency_loss += args.fluency*Fluency_Loss
        train_fluency_loss = train_fluency_loss.detach()
        train_loss += Total_Loss
        train_loss = train_loss.detach()

       
        Total_Loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        generator_model.zero_grad()
        classifier_model.zero_grad()
        
        predictions = torch.argmax(classifier_output.logits, dim=-1)
 
    #print(train_table)
        
    train_metrics = {"Directional Loss (Training)": train_directional_loss/len(train_dataloader),
                    "Fluency Loss (Training)": train_fluency_loss/len(train_dataloader),
                    "Total Loss (Training)": train_loss/len(train_dataloader)}
    

    # Output with sampling
    output_sampling = generate_sampling(batch_x)

    # Decoding Outputs for printing and storing
    output_decoded = generator_tokenizer.batch_decode(output_seq, skip_special_tokens=False)
    output_sampled_decoded = generator_tokenizer.batch_decode(output_sampling, skip_special_tokens=True)
    
    for i in range(len(input_decoded)):
        train_table.rows.append([input_decoded[i],output_decoded[i],output_sampled_decoded[i]])
    print()
    print(train_table)



    print("\nValidation\n")

    val_directional_loss = 0
    val_fluency_loss = 0
    val_loss = 0
    
    val_table = BeautifulTable(maxwidth=150)
    val_table.rows.append(["Input", "Output Greedy", "Output Sampling"])

    for batch in val_dataloader:
        input_decoded = batch['x']
        batch_x = generator_tokenizer(batch['x'], padding=True, truncation=True)
        labels = batch['y']

        with torch.no_grad():
            classifier_embeddings, output_seq, Fluency_Loss, attention_mask_Output  = generate(batch_x)
            classifier_output = classifier_model(inputs_embeds=classifier_embeddings, attention_mask=attention_mask_Output, labels=labels)
            Directional_Loss = classifier_output.loss  
            Total_Loss = Directional_Loss + args.fluency*Fluency_Loss

        
        # Saving all 3 losses
        val_directional_loss += Directional_Loss
        val_directional_loss = val_directional_loss.detach()
        val_fluency_loss += args.fluency*Fluency_Loss
        val_fluency_loss = val_fluency_loss.detach()
        val_loss += Total_Loss
        val_loss = val_loss.detach()

        
        predictions = torch.argmax(classifier_output.logits, dim=-1)
 
    #print(train_table)
        
    val_metrics = {"Directional Loss (Validation)": val_directional_loss/len(val_dataloader),
                    "Fluency Loss (Validation)": val_fluency_loss/len(val_dataloader),
                    "Total Loss (Validation)": val_loss/len(val_dataloader)}
    

    # Output with sampling
    output_sampling = generate_sampling(batch_x)

    # Decoding Outputs for printing and storing
    output_decoded = generator_tokenizer.batch_decode(output_seq, skip_special_tokens=False)
    output_sampled_decoded = generator_tokenizer.batch_decode(output_sampling, skip_special_tokens=True)
    
    for i in range(len(input_decoded)):
        val_table.rows.append([input_decoded[i],output_decoded[i],output_sampled_decoded[i]])
    print()
    print(val_table)



    print("\nEpoch time:",time.time()-start_time)

    print("\n\n===========================================================================\n\n")
    
    scheduler.step()
    
    wandb.log({**train_metrics, **val_metrics})


exit()

