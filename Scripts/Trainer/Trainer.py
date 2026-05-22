import torch
import torch.nn as nn
import torch.optim as optim
import os
import torch.utils.data.dataloader as dataloader
from tqdm import tqdm
from sklearn.metrics import f1_score, recall_score
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

class ModelTrainer:
    def __init__(self, model, device, loss_fn,learning_rate, batch_size,sampler):
        super(ModelTrainer,self).__init__()
        
        #Intialize variables for training and evaluation
        self.optimizer = None
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.start_epoch = 0
        self.sampler = sampler

        self.train_loader = None
        self.test_loader = None
        self.val_loader = None

        self.train_loss = []
        self.test_loss = []
        self.train_acc = []
        self.test_acc = []

        self.set_optimizer()

        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, 
                                                     T_0=10,
                                                     T_mult=2,
                                                     eta_min=1e-6)

    #Set optimizer
    def set_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate,weight_decay=1e-4)
    
    #Set data loader for test and train
    def set_data(self, train_set, test_set, val_set):
        print(f"Number of training examples: {len(train_set)}")
        print(f"Number of testing examples: {len(test_set)}")
        print(f"Number of Val examples: {len(val_set)}")


        self.train_loader = dataloader.DataLoader(train_set, batch_size=self.batch_size,shuffle=True, num_workers=0, pin_memory=True)
        self.test_loader = dataloader.DataLoader(test_set, batch_size=self.batch_size, shuffle=False, num_workers=0)
        self.val_loader = dataloader.DataLoader(val_set, batch_size=self.batch_size, shuffle=False, num_workers=0)

    #This function will perform single training epoch using our training data
    def Training(self):

        #Check for if training dataset is available
        if self.train_loader == None:
            print("Training Dataset loader not set")
        
        self.model.to(self.device)
        epoch_train_loss = 0
        #Train
        self.model.train()
        for i, (X,Y) in enumerate(tqdm(self.train_loader, leave=False, desc="Training")):
            #Set X and Y to device
            X,Y = X.to(self.device), Y.to(self.device)

            X = X + torch.randn_like(X)*0.005

            #Forward pass
            y_pred = self.model(X)

            #Set Loss
            loss = self.loss_fn(y_pred, Y.unsqueeze(1))
            
            #Zero out gradient
            self.optimizer.zero_grad()

            #Backpropogation and set gradient
            loss.backward()

            #Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            #Optimization
            self.optimizer.step()

            self.scheduler.step()

            #Keep track of train loss for plotting
            epoch_train_loss += loss.item()

        epoch_train_loss /= len(self.train_loader)

        self.train_loss.append(epoch_train_loss) 
        
    #Evaluation of model this runs per one epoch
    def eval_model(self, train_test_val = "test"):
        #Check for data loader
        if self.test_loader is None:
            print(f"No test loader available")
        
        self.model.to(self.device)

        loader = None 
        state = "Evaluating"
        if train_test_val == "test":
            loader = self.test_loader
            state += "test"
        elif train_test_val == "train":
            loader = self.train_loader
            state += "train"
        elif train_test_val == "val":
            loader = self.val_loader
            state += "val"
        else:
            ValueError("Invalid Dataset, train_test should be train/test")
        
        #Evaluation and Initialize variables
        epoc_acc = 0
        epoch_loss = 0
        correct_pred = 0
        sample = 0
        all_preds = []
        all_labels = []
        f1,recall = 0,0

        self.model.eval()
        with torch.inference_mode():
            for i, (X,Y) in enumerate(tqdm(loader,leave=False, desc=state)):
                #Set X, Y to device
                X,Y = X.to(self.device), Y.to(self.device)

                #Forward pass
                fx = self.model(X)

                #Loss
                loss = self.loss_fn(fx, Y.unsqueeze(1))
                epoch_loss += loss.item()

                #Log sum of acc for BCE
                preds = torch.sigmoid(fx)
                preds = (preds > 0.5).float()
                # epoc_acc += (preds.squeeze(1) == Y).sum().item()
                correct_pred += (preds.squeeze(1) == Y).sum().item()
                sample += Y.size(0)

                #Calculate preds and labels for F1 and recall score
                all_preds.extend(preds.squeeze(1).cpu().numpy().astype(int))
                all_labels.extend(Y.cpu().numpy().astype(int))
        
        epoc_acc = correct_pred/sample
        epoch_loss /=len(loader)
        self.all_preds = all_preds
        self.all_labels = all_labels
        
        #Log accuracy, loss, F1 and recall from the epoch
        if train_test_val == "train":
            self.train_acc.append(epoc_acc)
        elif train_test_val == "val":
            self.test_acc.append(epoc_acc)
            self.test_loss.append(epoch_loss)
            f1 = f1_score(self.all_labels,self.all_preds)
            recall = recall_score(self.all_labels,self.all_preds)
    
        return epoch_loss,epoc_acc,f1,recall
    
    def forward(self,x):
        return self.model(x)
        