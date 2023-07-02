from typing import List
import pandas as pd
import numpy as np
from utils_classifier import pre_process, max_sequence_length, create_dataloader, training, SentimentClassifier

from transformers import BertTokenizer

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
import torch.nn.functional as F


class Classifier:
    """
    The Classifier: complete the definition of this class template by providing a constructor (i.e. the
    __init__() function) and the 2 methods train() and predict() below. Please donot change
     """
    def __init__(self):
        self.num_classes = 3
        self.epochs = 8
        self.bs = 32
        self.lr = 8e-6
        self.max_length = 80
        self.PRE_TRAINED_MODEL = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.PRE_TRAINED_MODEL)

    ############################################# comp
    def train(self, train_filename: str, dev_filename: str, device: torch.device):
        """
        Trains the classifier model on the training set stored in file trainfile
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
          - DO NOT USE THE DEV DATA AS TRAINING EXAMPLES, YOU CAN USE THEM ONLY FOR THE OPTIMIZATION
         OF MODEL HYPERPARAMETERS
        """
        # Open files
        trainfile = pd.read_csv(train_filename, sep='\t', names = ["polarity", "category", "word", "offsets", "sentence"])
        devfile = pd.read_csv(dev_filename, sep='\t', names = ["polarity", "category", "word", "offsets", "sentence"])
        
        # Proprocess the file
        trainfile = pre_process(trainfile)
        devfile = pre_process(devfile)
        
        # Create dataloaders
        self.max_length = max_sequence_length(trainfile, self.tokenizer)       
        train_loader = create_dataloader(trainfile, self.tokenizer, self.max_length, self.bs)
        val_loader = create_dataloader(devfile, self.tokenizer, self.max_length, self.bs)

        # Create the model
        model = SentimentClassifier(self.num_classes)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        total_steps = len(train_loader) * self.epochs
        scheduler = lr_scheduler.LinearLR(optimizer, total_iters=total_steps)
        loss_function = nn.CrossEntropyLoss().to(device)
            
        # Training of the model
        training(self.epochs, model, train_loader, loss_function, optimizer, device, scheduler, trainfile, val_loader, devfile)


    def predict(self, data_filename: str, device: torch.device) -> List[str]:
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
        """

        # Open files
        datafile = pd.read_csv(data_filename, sep='\t', names = ["polarity", "category", "word", "offsets", "sentence"])
        
        # Preprocess the file
        datafile = pre_process(datafile)
        
        # Create dataloaders
        data_loader = create_dataloader(datafile, self.tokenizer, self.max_length, self.bs, shuffle=False)
        
        # Load the best model
        model = SentimentClassifier(self.num_classes).to(device)
        model.load_state_dict(torch.load('state_dict_best_model.pth'))

        # Set the model to evaluation mode
        model.eval()
        
        output_labels = []

        for row in data_loader:
            input_ids = row["input_ids"].to(device)
            attention_mask = row["attention_mask"].to(device)
            
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            logits = F.softmax(outputs, dim=1)
            logits = logits.detach().cpu().numpy()
            outputs = np.argmax(logits, axis=1)

            # convert labels in words
            for label in outputs:
                if label == 2:
                    output_labels.append('positive')
                
                elif label == 1:
                    output_labels.append('neutral')
                    
                elif label == 0:
                    output_labels.append('negative')

        return np.array(output_labels)