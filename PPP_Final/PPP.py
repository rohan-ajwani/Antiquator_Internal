import argparse

import logging

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
import evaluate

from beautifultable import BeautifulTable

parser = argparse.ArgumentParser()
parser.add_argument("--generator_model", type=str, default='gpt2-large')
parser.add_argument("--classifier_model", type=str, default='gpt2')
parser.add_argument("--prompt_length", type=int, default=10)
parser.add_argument("--fluency", type=float, default=0.2)
parser.add_argument("--max_new_tokens", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--n_epochs", type=int, default=20)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--desired_label", type=int, default=0)
parser.add_argument("--train_data_size", type=int, default=160)
parser.add_argument("--val_data_size", type=int, default=40)
args = parser.parse_args()
print(args)


perplexity = evaluate.load("perplexity", module_type="metric", use_cache=False)


if (args.desired_label==0):
    run_name = 'Negative_'+args.generator_model+'_'+args.classifier_model+'_'+str(args.prompt_length)+'prompt_'+str(args.fluency)+'fluency_'+str(args.max_new_tokens)+'tokens_'+str(args.learning_rate)+'lr_'+str(args.batch_size)+'bsz_'+str(args.n_epochs)+'epochs_'+str(args.train_data_size)+'_'+str(args.val_data_size)
else:
    run_name = 'Positive_'+args.generator_model+'_'+args.classifier_model+'_'+str(args.prompt_length)+'prompt_'+str(args.fluency)+'fluency_'+str(args.max_new_tokens)+'tokens_'+str(args.learning_rate)+'lr_'+str(args.batch_size)+'bsz_'+str(args.n_epochs)+'epochs_'+str(args.train_data_size)+'_'+str(args.val_data_size)



config = dict(generator_model=args.generator_model,
              classifier_model=args.classifier_model,
            prompt_length=args.prompt_length,
            fluency=args.fluency,
            max_new_tokens=args.max_new_tokens,
            learning_rate=args.learning_rate,
            gamma_decay=0.95,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size)
              

wandb.login()


project_name = 'Plug n Play'


wandb.init(project=project_name, name=run_name, config=config)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    

### GET DATASET ###

data_list = []

df = pd.read_csv('PPP_DATASET.csv')
text_set = df['Openings'].values.tolist()

print("Complete Dataset Size:",len(text_set))

for i in range(len(text_set)):
    data_list.append([text_set[i], args.desired_label])
sentiment_df = pd.DataFrame(data_list, columns = ['Text', 'Label'])


#Remove Null
sentiment_df['Text'].replace('', np.nan, inplace=True)
sentiment_df = sentiment_df.dropna()

print("Dataset Size:",len(sentiment_df))

print(sentiment_df)

remaining_dataset, train_dataset = train_test_split(sentiment_df, test_size=args.train_data_size, random_state=0)
if (len(remaining_dataset) <= args.val_data_size):
    val_dataset = remaining_dataset
else:
    _, val_dataset = train_test_split(remaining_dataset, test_size=args.val_data_size, random_state=0)
print("Train Dataset Size:",len(train_dataset))
print("Validation Dataset Size:",len(val_dataset))

train_dataset_text = train_dataset['Text'].tolist()
train_dataset_label = train_dataset['Label'].tolist()
val_dataset_text = val_dataset['Text'].tolist()
val_dataset_label = val_dataset['Label'].tolist()

print("Train Label sum =",sum(train_dataset_label))
print("Validation Label sum =",sum(val_dataset_label))


#Dataset dictionary
class SentimentDataset(Dataset):

    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        x = self.data[index]
        y = self.label[index]
        return_map={"x": x, "y": y}
        return return_map

    def __len__(self):
        return len(self.label)

train_dataset_dict = SentimentDataset(train_dataset_text, train_dataset_label)
val_dataset_dict = SentimentDataset(val_dataset_text, val_dataset_label)

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
generator_tokenizer.padding_side = 'left'

generator_model = GPT2LMHeadModel.from_pretrained(args.generator_model).to(device)

print("Loaded Generator Model")


### GENERATOR CREATED ###


### GET CLASSIFIER ###


classifier_checkpoint = args.classifier_checkpoint
classifier_tokenizer = GPT2Tokenizer.from_pretrained(args.classifier_model)
classifier_tokenizer.padding_side = 'left'

classifier_model = AutoModelForSequenceClassification.from_pretrained(args.classifier_model, num_labels=2, pad_token_id=classifier_tokenizer.eos_token_id).to(device)
print("Loaded Classifier Model")

checkpoint = torch.load(classifier_checkpoint, map_location=device)
for key in list(checkpoint.keys()):
    if 'model.' in key:
        checkpoint[key.replace('model.', '')] = checkpoint[key]
        del checkpoint[key]
    if 'module.' in key:
        checkpoint[key.replace('module.', '')] = checkpoint[key]
        del checkpoint[key]

classifier_model.load_state_dict(checkpoint)

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
## 2. Find n-grams

def CategoricalCrossEntropy(output_logits, target_logits):  #output_logits are of shape [bsz,vocab], targets are of shape [bsz], attention_slice is of size [bsz]

    bsz = output_logits.shape[0]

    target_probs = torch.exp(nn.LogSoftmax(dim=1)(target_logits))
    
    criterion = nn.CrossEntropyLoss(reduction='none')

    CCE = criterion(output_logits,target_probs)     #returns [bsz]
    
    CCE_Final = torch.sum(CCE)/bsz

    return CCE_Final



def get_ngrams(sentence, n):
    words = sentence.split()
    ngrams = set()

    if n <= 0 or n > len(words):
        return ngrams

    for i in range(len(words) - n + 1):
        ngram = " ".join(words[i:i+n])
        ngrams.add(ngram)

    return ngrams

### HELPER FUNCTION DONE ###



### OPTIMIZER & SCHEDULER ###

optimizer = torch.optim.AdamW([prompt_embeddings], lr=args.learning_rate)

print("Optimizer:",optimizer)

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)

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

    temperature = 0.1

    max_new_tokens = args.max_new_tokens
    
    
    attention_mask_withPrompt = torch.cat([torch.ones((bsz, prompt_length)).to(device), attention_mask_withoutPrompt], dim=1)  #-> attention mask for prompt + source

    #print("Input")
    #print(generator_tokenizer.batch_decode(input_ids, skip_special_tokens=True))
    
    output_seq = input_ids
    output_seq_withoutPrompt = input_ids

    source_embeddings = generator_model.get_input_embeddings()(input_ids).to(device)
    classifier_embeddings = classifier_model.get_input_embeddings()(input_ids).to(device)
    
    embeddings_matrix_generator = generator_model.get_input_embeddings().weight[:V,:]
    embeddings_matrix_classifier = classifier_model.get_input_embeddings().weight[:V,:]

    prompt_embeddings_broadcasted = torch.stack([prompt_embeddings]*bsz)
    
    #print(max_new_tokens)
    
    Fluency_Loss = 0
    
    for generate_step in range(max_new_tokens):
        
        if (generate_step == 0):
            outputs_withoutPrompt = generator_model(inputs_embeds=source_embeddings, attention_mask=attention_mask_withoutPrompt, return_dict=True)
            outputs_withPrompt = generator_model(inputs_embeds=torch.concat([prompt_embeddings_broadcasted, source_embeddings], dim=1), attention_mask=attention_mask_withPrompt, return_dict=True) 
        else:
            outputs_withoutPrompt = generator_model(inputs_embeds=input_embeddings_withoutPrompt, attention_mask=attention_mask_withoutPrompt, past_key_values=past_key_values_withoutPrompt, return_dict=True)
            outputs_withPrompt = generator_model(inputs_embeds=input_embeddings, attention_mask=attention_mask_withPrompt, past_key_values=past_key_values_withPrompt, return_dict=True)
            
        logits_withoutPrompt = outputs_withoutPrompt.logits[:, -1, :V]      #outputs_withoutPrompt.logits[:, -1, :V]
        logits_withPrompt = outputs_withPrompt.logits[:, -1, :V]    #outputs_withPrompt.logits[:, -1, :V] -> using [0] instead of .logits as return_dict=False

        past_key_values_withoutPrompt = outputs_withoutPrompt.past_key_values
        past_key_values_withPrompt = outputs_withPrompt.past_key_values
            
        loss = CategoricalCrossEntropy(output_logits=logits_withPrompt , target_logits=logits_withoutPrompt)    # -------- Loss Component 1

        next_tokens = torch.argmax(logits_withPrompt, dim=1)  # (bsz) Note: No gradients here
        next_tokens = next_tokens.unsqueeze(1)  # (bsz,1)
        output_seq = torch.concat([output_seq,next_tokens], dim=1)  # (bsz,S+L+1)

        
        softmax_logits_withPrompt = torch.exp(nn.LogSoftmax(dim=1)(logits_withPrompt/temperature))

        next_tokens_withoutPrompt = torch.argmax(logits_withPrompt, dim=1)  # (bsz) Note: No gradients here
        next_tokens_withoutPrompt = next_tokens_withoutPrompt.unsqueeze(1)  # (bsz,1)
        output_seq_withoutPrompt = torch.concat([output_seq_withoutPrompt,next_tokens_withoutPrompt], dim=1)  # (bsz,S+L+1)

        next_token_embeddings_generator = torch.matmul(softmax_logits_withPrompt, embeddings_matrix_generator)  # (bsz, D_gen)
        input_embeddings = next_token_embeddings_generator.unsqueeze(1)



        next_token_embeddings_generator_withoutPrompt = torch.matmul(softmax_logits_withPrompt, embeddings_matrix_generator)  # (bsz, D_gen)
        input_embeddings_withoutPrompt = next_token_embeddings_generator_withoutPrompt.unsqueeze(1)   #(bsz, 1, D_gen)


        next_token_embeddings_classifier = torch.matmul(softmax_logits_withPrompt, embeddings_matrix_classifier)  # (bsz, D_clf)

        if (classifier_embeddings == []):
            classifier_embeddings.append(next_token_embeddings_classifier.unsqueeze(1))
            classifier_embeddings = torch.stack(classifier_embeddings).to(device)
            classifier_embeddings = classifier_embeddings.squeeze(0)
        else:
            classifier_embeddings = torch.concat([classifier_embeddings, next_token_embeddings_classifier.unsqueeze(1)], dim=1)  # (bsz, L+1, D_clf)


        attention_mask_withPrompt = torch.concat([attention_mask_withPrompt,torch.tensor([[1]]*bsz)], dim=1)
        attention_mask_withoutPrompt = torch.concat([attention_mask_withoutPrompt,torch.tensor([[1]]*bsz)], dim=1)

        
        #input_embeddings =  generator_model.get_input_embeddings()(next_tokens).to(device)
        Fluency_Loss += loss

        del outputs_withPrompt
        gc.collect()
        torch.cuda.empty_cache()


    Fluency_Loss = Fluency_Loss/max_new_tokens
    
    return classifier_embeddings, output_seq, output_seq_withoutPrompt, Fluency_Loss, attention_mask_withoutPrompt   #classifier needs output attention mask to disregard tokens produced after sequence length




### GENERATE WITH SAMPLING ###

def generate_sampling(batch_x):

    input_ids = torch.tensor(batch_x['input_ids']).to(device)
    attention_mask_withoutPrompt = torch.tensor(batch_x['attention_mask']).to(device)
    bsz, L = input_ids.shape
    V = 50257

    max_new_tokens = args.max_new_tokens


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
        probs = torch.exp(nn.LogSoftmax(dim=1)(logits))

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
    train_table.rows.append(["Input", "Output Greedy", "Output Sampled"])

    for batch in train_dataloader:
        input_decoded = batch['x']
        batch_x = generator_tokenizer(batch['x'], padding=True, truncation=True)
        labels = batch['y']

        classifier_embeddings, output_seq, output_seq_withoutPrompt, Fluency_Loss, attention_mask_Output  = generate(batch_x)
        
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

        if (epoch>0):       
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
    #output_sampling = generate_sampling(batch_x)

    # Decoding Outputs for printing and storing
    output_decoded = generator_tokenizer.batch_decode(output_seq, skip_special_tokens=False)
    output_sampled_decoded = generator_tokenizer.batch_decode(output_seq_withoutPrompt, skip_special_tokens=True)
    
    for i in range(len(input_decoded)):
        train_table.rows.append([input_decoded[i],output_decoded[i],output_sampled_decoded[i]])
    print()
    print(train_table)

    
    greedy_perplexity = 0
    sampled_perplexity = 0
    dist_1_greedy=0
    dist_2_greedy=0
    dist_3_greedy=0
    dist_1_sampled=0
    dist_2_sampled=0
    dist_3_sampled=0

    print("\nValidation\n")

    val_directional_loss = 0
    val_fluency_loss = 0
    val_loss = 0
    val_correct = 0

    for batch in val_dataloader:
        input_decoded = batch['x']
        batch_x = generator_tokenizer(batch['x'], padding=True, truncation=True)
        labels = batch['y']

        with torch.no_grad():
            classifier_embeddings, output_seq, output_seq_withoutPrompt, Fluency_Loss, attention_mask_Output = generate(batch_x)
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
        
        #print("predictions.shape",predictions.shape)
        #print(predictions)
        #print(sum(predictions==args.desired_label))
    
        val_correct += sum(predictions==args.desired_label)
        
        output_sampling = generate_sampling(batch_x)

        # Decoding Outputs for calculating perplexity, dist-1,2,3
        output_decoded = generator_tokenizer.batch_decode(output_seq, skip_special_tokens=True)
        output_sampled_decoded = generator_tokenizer.batch_decode(output_sampling, skip_special_tokens=True)
        
        greedy_perplexity += perplexity.compute(model_id=args.generator_model, add_start_token=False, predictions=output_decoded)['mean_perplexity']/len(val_dataloader)

        
        for i in range(len(output_decoded)):
            words = len(output_decoded[i].split(' '))

            n_grams_1 = get_ngrams(output_decoded[i],1)
            n_grams_2 = get_ngrams(output_decoded[i],2)
            n_grams_3 = get_ngrams(output_decoded[i],3)

            dist_1_greedy += (len(n_grams_1)/words)/args.val_data_size
            dist_2_greedy += (len(n_grams_2)/(words-1))/args.val_data_size
            dist_3_greedy += (len(n_grams_3)/(words-2))/args.val_data_size


        
    val_metrics = {"Correctly classified (Val)": val_correct}
                    "Greedy Perplexity (Val)":greedy_perplexity,
                    "Greedy Dist-1 (Val)":dist_1_greedy,
                    "Greedy Dist-2 (Val)":dist_2_greedy,
                    "Greedy Dist-3 (Val)":dist_3_greedy}

 

    
    for i in range(len(input_decoded)):
        val_table.rows.append([input_decoded[i],output_decoded[i],output_sampled_decoded[i]])
    print()
    print(val_table)

    print("Correct Predictions:",val_correct)

    print("\nEpoch time:",time.time()-start_time)

    print("\n\n===========================================================================\n\n")
    
    scheduler.step()
    
    wandb.log({**train_metrics, **val_metrics})


exit()

