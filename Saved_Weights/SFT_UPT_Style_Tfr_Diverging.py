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
parser.add_argument("--train_data_size", type=int, default=32000)
parser.add_argument("--val_data_size", type=int, default=8000)
parser.add_argument("--generator_weights_path", type=str, default="/checkpoint/ajwaniro/Identity_GPT2_Checkpoints/gpt2-large_3e-05lr_32bsz_2epochs")   #Note: generator loaded directly as it was saved using HF trainer
parser.add_argument("--classifier_weights_path", type=str, default="/checkpoint/ajwaniro/Classifier_Checkpoints_GPT2/Yelp_Dataset/gpt2_1e-05lr_32bsz_2epochs.pt")
parser.add_argument("--prompt_length", type=int, default=20)
parser.add_argument("--prompt_weight", type=float, default=0.1)
parser.add_argument("--fluency_loss_weight", type=float, default=0.2)
parser.add_argument("--gumbel_weight", type=float, default=0.0)
parser.add_argument("--optimizer", type=str, default="ADAMW")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--n_epochs", type=int, default=5)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--gamma", type=float, default=1.0) # LR Decay exponential factor
parser.add_argument("--desired_label", type=int, default=0)
parser.add_argument("--top_k", type=int, default=30)
parser.add_argument("--sample_temperature", type=float, default=0.95)
parser.add_argument("--seed", type=int, default=42, help="Seed for everything")
args = parser.parse_args()
print(args)

run_name = 'gpt2large_'+str(args.prompt_length)+'Prompt_'+str(args.prompt_weight)+'PWT_'+str(args.fluency_loss_weight)+'Fluency_'+str(args.gumbel_weight)+'Gumbel_'+str(args.learning_rate)+'LR_'+str(args.gamma)+'Gamma_'+args.optimizer+'_'+str(args.batch_size)+'BSZ_'+str(args.n_epochs)+'Epochs_'+str(args.train_data_size)+'_'+str(args.val_data_size)

config = dict(prompt_length=args.prompt_length,
            prompt_weight=args.prompt_weight,
            fluency_loss_weight=args.fluency_loss_weight,
            gumbel_weight=args.gumbel_weight,
            learning_rate=args.learning_rate,
            gamma_decay=args.gamma,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            optimizer=args.optimizer,
            desired_label=args.desired_label)
              

wandb.login()


torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

if (args.desired_label == 0):
    save_dir = '/checkpoint/ajwaniro/StylTfr_SFT_UPT/Sentiment/Yelp/Negative/' + run_name
    project_name = 'StylTfr SFT+UPT Pos2Neg'
else:
    save_dir = '/checkpoint/ajwaniro/StylTfr_SFT_UPT/Sentiment/Yelp/Positive/' + run_name
    project_name = 'StylTfr SFT+UPT Neg2Pos'
    
training_state_file_path = save_dir + '/training_state.pth.tar'


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
    text = item.strip()+" = "
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


print("Device: ",device)

train_dataloader = DataLoader(train_dataset_dict, generator=torch.Generator(device=device), batch_size=config['batch_size'], shuffle=True)
val_dataloader = DataLoader(val_dataset_dict, generator=torch.Generator(device=device), batch_size=config['batch_size'], shuffle=True)          

print("Train DataLoader Size:",len(train_dataloader))
print("Validation DataLoader Size:",len(val_dataloader))
    

### DATASET COMPLETE, DATALOADERS CREATED###



### GET CLASSIFIER ###

classifier_checkpoint = args.classifier_weights_path

classifier_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

classifier_model = AutoModelForSequenceClassification.from_pretrained('gpt2', num_labels=2, pad_token_id=classifier_tokenizer.eos_token_id)
print("Loaded Classifier Model")

classifier_model.load_state_dict(torch.load(classifier_checkpoint, map_location=torch.device('cpu')))
print("Loaded Classifier Checkpoints")

classifier_model = classifier_model.to(device)

### CLASSIFIER WEIGHTS LOADED ###



### GET GENERATOR ###

generator_tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
generator_tokenizer.pad_token = generator_tokenizer.eos_token

generator_checkpoint = args.generator_weights_path

generator_model = GPT2LMHeadModel.from_pretrained(generator_checkpoint).to(device)

print("Loaded Generator Model")


### GENERATOR CREATED ###


### INITIALIZE PROMPT EMBEDDINGS ###

embedding_dim = generator_model.transformer.wte(torch.tensor(0)).detach().shape[-1]

#if args.custom_prompt == "":
prompt_length = args.prompt_length
embed = torch.randn((prompt_length,embedding_dim), requires_grad=False) * args.prompt_weight
prompt_embeddings = embed.requires_grad_(True)
#else:
#    prompt_tokenized = generator_tokenizer(args.custom_prompt)
#    prompt_tokenized = torch.tensor(prompt_tokenized['input_ids'])
#    prompt_length = prompt_tokenized.shape[-1]
#    embed = generator_model.get_input_embeddings()(prompt_tokenized).detach().clone().detach() * args.prompt_weight
#    prompt_embeddings = embed.requires_grad_(True)

#print("Prompt Embeddings Initialized")
### PROMPT EMBEDDINGS INITIALIZED ###


### HELPER FUNCTIONS ###
## 1. CCE
## 2. Gumbel Noise

def CategoricalCrossEntropy(output_logits, target_logits, attention_slice):  #output and target logits are of shape [bsz,vocab], attention_slice is of size [bsz]

    bsz = output_logits.shape[0]
    
    output_logprobs = nn.LogSoftmax(dim=1)(output_logits)
    target_probs = torch.exp(nn.LogSoftmax(dim=1)(target_logits))

    CCE = -torch.sum(torch.mul(target_probs, output_logprobs),dim=-1)   # elementwise mult->[bsz,vocab], sum(dim=-1)->[bsz]
    CCE_masked = torch.mul(CCE, attention_slice)    # elementwise mult [bsz]*[bsz] -> [bsz], masked with attention, 0 for sequences deemed to be completed
    CCE_Final = torch.sum(CCE_masked)/(torch.sum(attention_slice)+1e-20)  #denominator only sequences whose CCE is accounted, so don't divide by bsz
    
    return CCE_Final


def gumbel_noise(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)

### HELPER FUNCTION DONE ###


### OPTIMIZER & SCHEDULER ###

if (args.optimizer == "SGD"):
    optimizer = torch.optim.SGD([prompt_embeddings], lr=args.learning_rate)
elif (args.optimizer == "SGDM"):
    optimizer = torch.optim.SGD([prompt_embeddings], lr=args.learning_rate, momentum=0.9)
elif (args.optimizer == "ADAM"):
    optimizer = torch.optim.Adam([prompt_embeddings], lr=args.learning_rate)
elif (args.optimizer == "ADAMW"):
    optimizer = torch.optim.AdamW([prompt_embeddings], lr=args.learning_rate)

print("Optimizer:",optimizer)

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.gamma)

print("Scheduler:",scheduler)
print()



### GENERATE AND FIND LOSS ###

def generate(batch_x):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    """

    temperature = 0.05
    
    input_ids = torch.tensor(batch_x['input_ids'])
    attention_mask_withoutPrompt = torch.tensor(batch_x['attention_mask'])
    bsz, L = input_ids.shape
    V = 50257

    max_new_tokens = L-2    # 2 tokens to account for " . ="
    #max_new_tokens = min(max_new_tokens, 15)
    attention_mask_Output = attention_mask_withoutPrompt[:,L-max_new_tokens:]    # as output length = L-2 
    
    prompt_embeddings_broadcasted = torch.stack([prompt_embeddings]*bsz)
    
    attention_mask_withPrompt = torch.cat([torch.ones((bsz, prompt_length)).to(device), attention_mask_withoutPrompt], dim=1)  #-> attention mask for prompt + source
    
    #print("Input")
    #print(generator_tokenizer.batch_decode(input_ids, skip_special_tokens=True))
    
    output_seq = input_ids
    output_seq_withoutPrompt = input_ids


    source_embeddings = generator_model.get_input_embeddings()(input_ids).to(device)
    #classifier_embeddings = classifier_model.get_input_embeddings()(input_ids).to(device)
    classifier_embeddings = []  # fresh embeddings after "input . =" passed to classifier

    embeddings_matrix_generator = generator_model.get_input_embeddings().weight[:V,:]
    embeddings_matrix_classifier = classifier_model.get_input_embeddings().weight[:V,:]
    
    #print(max_new_tokens)
    
    Fluency_Loss = 0
    


    for generate_step in range(max_new_tokens):
        
        if (generate_step == 0):
            outputs_withoutPrompt = generator_model(inputs_embeds=source_embeddings, attention_mask=attention_mask_withoutPrompt, return_dict=True)
            outputs_withPrompt = generator_model(inputs_embeds=torch.concat([prompt_embeddings_broadcasted ,source_embeddings], dim=1), attention_mask=attention_mask_withPrompt,return_dict=True)
            
        else:
            outputs_withoutPrompt = generator_model(inputs_embeds=input_embeddings_withoutPrompt, attention_mask=attention_mask_withoutPrompt, past_key_values=past_key_values_withoutPrompt, return_dict=True)
            outputs_withPrompt = generator_model(inputs_embeds=input_embeddings_withPrompt, attention_mask=attention_mask_withPrompt, past_key_values=past_key_values_withPrompt, return_dict=True)
            
            
        logits_withoutPrompt = outputs_withoutPrompt.logits[:, -1, :V]      #outputs_withoutPrompt.logits[:, -1, :V]
        logits_withPrompt = outputs_withPrompt.logits[:, -1, :V]    #outputs_withPrompt.logits[:, -1, :V] -> using [0] instead of .logits as return_dict=False
        
        past_key_values_withoutPrompt = outputs_withoutPrompt.past_key_values
        past_key_values_withPrompt = outputs_withPrompt.past_key_values
            
        Fluency_Loss += CategoricalCrossEntropy(output_logits=logits_withPrompt , target_logits=logits_withoutPrompt, attention_slice=attention_mask_Output[:,generate_step])    # -------- Loss Component 1

        #adding weighted gumbel noise
        gumbel_logits_withPrompt = logits_withPrompt + args.gumbel_weight*gumbel_noise(logits_withPrompt.shape)

        next_tokens = torch.argmax(gumbel_logits_withPrompt, dim=1)  # (bsz) Note: No gradients here
        next_tokens = next_tokens.unsqueeze(1)  # (bsz,1)
        output_seq = torch.concat([output_seq,next_tokens], dim=1)  # (bsz,S+L+1)

        next_tokens_withoutPrompt = torch.argmax(logits_withoutPrompt, dim=1)  # (bsz) Note: No gradients here
        next_tokens_withoutPrompt = next_tokens_withoutPrompt.unsqueeze(1)  # (bsz,1)
        output_seq_withoutPrompt = torch.concat([output_seq_withoutPrompt,next_tokens_withoutPrompt], dim=1)  # (bsz,S+L+1)



        softmax_logits_withPrompt = torch.exp(nn.LogSoftmax(dim=1)(gumbel_logits_withPrompt/temperature))

        next_token_embeddings_generator_withPrompt = torch.matmul(softmax_logits_withPrompt, embeddings_matrix_generator)  # (bsz, D_gen)
        input_embeddings_withPrompt = next_token_embeddings_generator_withPrompt.unsqueeze(1)   #(bsz, 1, D_gen)

        
        softmax_logits_withoutPrompt = torch.exp(nn.LogSoftmax(dim=1)(logits_withoutPrompt/temperature))

        next_token_embeddings_generator_withoutPrompt = torch.matmul(softmax_logits_withoutPrompt, embeddings_matrix_generator)  # (bsz, D_gen)
        input_embeddings_withoutPrompt = next_token_embeddings_generator_withoutPrompt.unsqueeze(1)   #(bsz, 1, D_gen)

        

        next_token_embeddings_classifier = torch.matmul(softmax_logits_withPrompt, embeddings_matrix_classifier)  # (bsz, D_clf)

        if (classifier_embeddings == []):
            classifier_embeddings.append(next_token_embeddings_classifier.unsqueeze(1))
            classifier_embeddings = torch.stack(classifier_embeddings).to(device)
            classifier_embeddings = classifier_embeddings.squeeze(0)
        else:
            classifier_embeddings = torch.concat([classifier_embeddings, next_token_embeddings_classifier.unsqueeze(1)], dim=1)  # (bsz, L+1, D_clf)


        attention_mask_withoutPrompt = torch.concat([attention_mask_withoutPrompt,torch.tensor([[1]]*bsz)], dim=1)
        attention_mask_withPrompt = torch.concat([attention_mask_withPrompt,torch.tensor([[1]]*bsz)], dim=1)

        
        del outputs_withoutPrompt
        gc.collect()
        torch.cuda.empty_cache()
        del outputs_withPrompt
        gc.collect()
        torch.cuda.empty_cache()

    Fluency_Loss = Fluency_Loss/max_new_tokens   # ---------------------------------------------- Loss Component 1
        
    return classifier_embeddings, output_seq, output_seq_withoutPrompt, Fluency_Loss, attention_mask_Output   #classifier needs output attention mask to disregard tokens produced after sequence length



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

        v, _ = torch.topk(logits, args.top_k)   #restrict vocab to top_k
        logits[logits < v[:, [-1]]] = -float('Inf')

        # getting prob of top k tokens using softmax (prob of other tokens are zero as logits are set to -Inf)
        probs = torch.exp(nn.LogSoftmax(dim=1)(logits/args.sample_temperature))
        
        #get next token from probabilities using sampling
        next_tokens = torch.multinomial(probs, num_samples=1) # (bsz,1)

        output_seq = torch.concat([output_seq,next_tokens], dim=1)

        attention_mask_withPrompt = torch.concat([attention_mask_withPrompt,torch.tensor([[1]]*bsz)], dim=1)

    return output_seq




### CHECKPOINT ###

current_epoch = 0

if os.path.exists(save_dir):
    
    print("Loading Saved Checkpoint")
    
    checkpoint = torch.load(training_state_file_path)

    # Load Epoch
    current_epoch = checkpoint['epoch']
    
    # Load Prompt Embeddings
    prompt_embeddings = checkpoint['prompt_embeddings']
    
    # Load Optimizer state
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    # Load Scheduler state
    scheduler.load_state_dict(checkpoint['scheduler'])
    

    
### TRAIN AND VALIDATE ###

for epoch in range(current_epoch, args.n_epochs):
    
    step = 0

    start_time = time.time()

    print("\nE P O C H :", epoch)
    
    train_directional_loss = 0
    train_fluency_loss = 0
    train_loss = 0
    
    print("\nTraining\n")
    
    train_table = BeautifulTable(maxwidth=100)
    train_table.rows.append(["Input", "Output Prompted (Greedy)", "Output Unprompted"])

    for batch in train_dataloader:
        input_decoded = batch['x']
        batch_x = generator_tokenizer(batch['x'], padding=True, truncation=True)
        labels = batch['y']

        classifier_embeddings, output_seq, output_seq_withoutPrompt, Fluency_Loss, attention_mask_Output = generate(batch_x)
        
        # Fluency loss is divided by number of tokens inside generate function --------------------------- Loss Component 1
        
        #print("Classifier Embeddings Shape:",classifier_embeddings.shape)
        
        # Classification Loss
        classifier_output = classifier_model(inputs_embeds=classifier_embeddings, attention_mask=attention_mask_Output, labels=labels)
        
        Directional_Loss = classifier_output.loss   # ---------------------------------------------------- Loss Component 2
        
        Total_Loss = Directional_Loss + args.fluency_loss_weight*Fluency_Loss   # ------------------------ Final Loss
        
        # Saving all 3 losses
        train_directional_loss += Directional_Loss
        train_directional_loss = train_directional_loss.detach()
        train_fluency_loss += args.fluency_loss_weight*Fluency_Loss
        train_fluency_loss = train_fluency_loss.detach()
        train_loss += Total_Loss
        train_loss = train_loss.detach()
       

        Total_Loss.backward()
        
       # print("Classifier Output Logits:",classifier_output.logits)
       # print()
       # print("prompt_embeddings Gradients:")
       # print(prompt_embeddings.grad)

        optimizer.step()
        optimizer.zero_grad()
        generator_model.zero_grad()
        classifier_model.zero_grad()
        
        predictions = torch.argmax(classifier_output.logits, dim=-1)
        
        
        # Output with sampling
        #output_sampling = generate_sampling(batch_x)
        
        
        # Decoding Outputs for printing and storing
        
        #input_decoded = batch['x']
        #output_decoded = generator_tokenizer.batch_decode(output_seq, skip_special_tokens=True)
        #output_sampled_decoded = generator_tokenizer.batch_decode(output_sampling, skip_special_tokens=True)
        
        #for i in range(len(input_decoded)):
        #    train_table.rows.append([input_decoded[i],output_decoded[i], output_sampled_decoded[i]])
 
    #print(train_table)
        
    print()
    print("Train Loss Total:",train_loss)
    train_metrics = {"Directional Loss (Training)": train_directional_loss/len(train_dataloader),
                    "Fluency Loss (Training)": train_fluency_loss/len(train_dataloader),
                    "Total Loss (Training)": train_loss/len(train_dataloader)}
    

    # Output with sampling
    #output_sampling = generate_sampling(batch_x)


    # Decoding Outputs for printing and storing

    output_decoded = generator_tokenizer.batch_decode(output_seq, skip_special_tokens=False)
    output_sampled_decoded = generator_tokenizer.batch_decode(output_seq_withoutPrompt, skip_special_tokens=False)

    for i in range(len(input_decoded)):
        train_table.rows.append([input_decoded[i],output_decoded[i], output_sampled_decoded[i]])
    print()
    print(train_table)



    print("\nValidation\n")

    val_table = BeautifulTable(maxwidth=100)
    val_table.rows.append(["Input", "Output Prompted (Greedy)", "Output Unprompted"])

    val_directional_loss = 0
    val_fluency_loss = 0
    val_loss = 0
    
    for batch in val_dataloader:
        input_decoded = batch['x']
        batch_x = generator_tokenizer(batch['x'], padding=True, truncation=True)
        labels = batch['y']
        
        with torch.no_grad():
            classifier_embeddings, output_seq, output_seq_withoutPrompt, Fluency_Loss, attention_mask_Output = generate(batch_x)
            classifier_output = classifier_model(inputs_embeds=classifier_embeddings, attention_mask=attention_mask_Output, labels=labels)
            Directional_Loss = classifier_output.loss
            val_Total_Loss = Directional_Loss + args.fluency_loss_weight*Fluency_Loss
            val_directional_loss += Directional_Loss
            val_fluency_loss += args.fluency_loss_weight*Fluency_Loss
            val_loss += val_Total_Loss
            
        # Output with sampling
        #output_sampling = generate_sampling(batch_x)
        
        
        # Decoding outputs
        
        #input_decoded = batch['x']
        #output_decoded = generator_tokenizer.batch_decode(output_seq, skip_special_tokens=True)
        #output_sampled_decoded = generator_tokenizer.batch_decode(output_sampling, skip_special_tokens=True)
        
        #for i in range(len(input_decoded)):
        #    val_table.rows.append([input_decoded[i],output_decoded[i], output_sampled_decoded[i]])
            
        
    #print(val_table)

    
    print()
    print("Val Loss Total:",val_loss)
    
    val_metrics = {"Directional Loss (Validation)": val_directional_loss/len(val_dataloader),
                    "Fluency Loss (Validation)": val_fluency_loss/len(val_dataloader),
                    "Total Loss (Validation)": val_loss/len(val_dataloader)}
    

    #Output with sampling
    #output_sampling = generate_sampling(batch_x)


    # Decoding outputs
    output_decoded = generator_tokenizer.batch_decode(output_seq, skip_special_tokens=False)
    output_sampled_decoded = generator_tokenizer.batch_decode(output_seq_withoutPrompt, skip_special_tokens=False)

    for i in range(len(input_decoded)):
        val_table.rows.append([input_decoded[i],output_decoded[i], output_sampled_decoded[i]])

    print()
    print(val_table)


    print("\nEpoch time:",time.time()-start_time)

    print("\n\n===========================================================================\n\n")
    
    scheduler.step()
    
    state = {'epoch':epoch+1, 'prompt_embeddings':prompt_embeddings, 'optimizer':optimizer.state_dict(), 'scheduler':scheduler.state_dict()}
    #if os.path.exists(save_dir):
    #    torch.save(state, training_state_file_path)
    #else:
    #    os.makedirs(save_dir)
    #    torch.save(state, training_state_file_path)
    
    wandb.log({**train_metrics, **val_metrics})


exit()

