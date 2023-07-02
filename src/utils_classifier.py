from transformers import BertModel
import torch
from tqdm import tqdm
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader

def aspect_categories(category):
  '''
  Change the category in more understandable sentence
  '''
  if category == 'AMBIENCE#GENERAL':
    return "Feeling about * ambience"
  elif category == 'FOOD#QUALITY' or category == 'DRINKS#QUALITY':
    return "Feeling about * quality"
  elif category == 'SERVICE#GENERAL':
    return "Feeling about * service"
  elif category == 'FOOD#STYLE_OPTIONS' or category == 'DRINKS#STYLE_OPTIONS':
    return "Feeling about * choices"
  elif category == 'RESTAURANT#MISCELLANEOUS' or category == 'RESTAURANT#GENERAL':
    return "Feeling about * restaurant"
  elif category == 'LOCATION#GENERAL':
    return 'Feeling about * location'
  elif category == 'RESTAURANT#PRICES' or category =='DRINKS#PRICES' or category == 'FOOD#PRICES':
    return 'Feeling about * prices'

def pre_process(df):
    '''
    tranform columns of dataframe df in usable information, ie numeric or sentences
    '''
    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    df["polarity"] = df["polarity"].apply(lambda x: label_map[x])
    df["category"] = df["category"].apply(lambda x: aspect_categories(x))   
    return df

def max_sequence_length(df, tokenizer):
    '''
    return the max length of tokenized sentences. Needed for encode_plus function
    '''
    max_lengths = 0
    for sentence in df["sentence"]:
        tokens = tokenizer.encode(sentence, max_length=1000)
        length = len(tokens)
        if length>max_lengths:
           max_lengths = length
    # Add 20% more space in case of longer sentences in devdata
    return int(1.2*max_lengths)

class create_dataset(Dataset):
    def __init__(self, polarities, categories, words, sentences, tokenizer, max_len_token):
        self.polarities = polarities
        self.categories = categories
        self.words = words
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_len_token = max_len_token
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, index):
        polarity = self.polarities[index]
        category = str(self.categories[index])
        word = str(self.words[index])
        sentence = str(self.sentences[index])
        aspect = category.replace("*", word)
        encoding = self.tokenizer.encode_plus(
            sentence,
            aspect,
            add_special_tokens=True,
            max_length=self.max_len_token,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        polarity = torch.tensor(polarity, dtype=torch.long)
        return {
            'sentence': sentence,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'polarities': polarity,
        }
    
def create_dataloader(df, tokenizer, max_len_token, batch_size, shuffle=True):
    '''
    create the dataset for training, with aspect+category tokenized, as well as sentences
    '''
    dataset = create_dataset(
        polarities = df["polarity"].to_numpy(), 
        categories = df["category"].to_numpy(), 
        words = df["word"].to_numpy(), 
        sentences = df["sentence"].to_numpy(),
        tokenizer = tokenizer,
        max_len_token = max_len_token
    )
    dataloader = DataLoader(
        dataset,
        batch_size= batch_size,
        shuffle=shuffle
    )
    return dataloader

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes=3):
        super(SentimentClassifier, self).__init__()
        PRE_TRAINED_MODEL = 'bert-base-uncased'
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL)
        self.drop = nn.Dropout(p=0.30)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
        input_ids = input_ids,
        attention_mask = attention_mask
        )[0:2]
        output = self.drop(pooled_output)
        return self.out(output)
    
def training(epochs, model, train_loader, loss_function, optimizer, device, scheduler, trainfile, val_loader, devfile):    
    '''
    train the model wrt. parameters given
    '''
    best_accuracy = 0
    for _ in tqdm(range(epochs)):
        #print(f'Epoch {epoch + 1}/{epochs}')
        train_acc, train_loss = training_helper(model, train_loader, loss_function, optimizer, device, scheduler, len(trainfile))
        
        #print(f'Train loss {train_loss} accuracy {train_acc}')
        
        if val_loader is not None:
            val_acc, val_loss = eval_function(model, val_loader, loss_function, device, len(devfile))
            #print(f'Val loss {val_loss} accuracy {val_acc}')
            #print()
        
        #keep the model with the best score on dev set
        if (val_acc > best_accuracy):
            torch.save(model.state_dict(), 'state_dict_best_model.pth')
            best_accuracy = val_acc

def training_helper(model, data_loader, loss_function, optimizer, device, scheduler, number_examples):
    '''
    train the model on one epoch
    '''
    
    model.train()
    losses = []
    correct_predictions = 0
    
    for row in data_loader:
        input_ids = row["input_ids"].to(device)
        attention_mask = row["attention_mask"].to(device)
        labels = row["polarities"].to(device)
                
        # compute the model
        outputs = model(
            input_ids = input_ids,
            attention_mask = attention_mask
        )
        #compute the loss and accuracy
        _, preds = torch.max(outputs, dim=1)
        loss = loss_function(outputs, labels)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())
        
        # Backpropagate
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
    return correct_predictions.double() / number_examples, np.mean(losses)

def eval_function(model, data_loader, loss_function, device, number_examples):
    '''
    evaluate the model on the dev set, and return the accuracy and loss
    '''
    
    model = model.eval()    
    losses = []
    correct_predictions = 0
    
    with torch.no_grad():
        for row in data_loader:
            input_ids = row["input_ids"].to(device)
            attention_mask = row["attention_mask"].to(device)
            labels = row["polarities"].to(device)
            # compute the model
            outputs = model(input_ids = input_ids, attention_mask = attention_mask)
            #compute loss function and accuracy
            _, preds = torch.max(outputs, dim=1)
            loss = loss_function(outputs, labels)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
            
    return correct_predictions.double() / number_examples, np.mean(losses)