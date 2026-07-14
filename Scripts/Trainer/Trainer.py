import torch
import torch.nn as nn
import torch.optim as optim
import os
import torch.utils.data.dataloader as dataloader
from tqdm import tqdm,trange
from sklearn.metrics import f1_score, recall_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import Augmentation
import time
from Resnet50 import CustomResnet50
import torch.nn.functional as F
from sklearn.metrics import roc_curve
import numpy as np

class ModelTrainer(nn.Module):
    def __init__(self, 
                 model, 
                 device, 
                 loss_fn,
                 batch_size,
                 sampler,
                 start_from_checkpoint=False,
                 teacher_present=False,
                 Temperature = 1.0,
                 alpha = 0.5,
                 distil_type = "hard",
                 optimal_threshold = None,
                 epochs = 0,
                 learning_rate=0.0,
                 optimizer = None):
        super(ModelTrainer,self).__init__()
        
        #Intialize variables for training and evaluation
        self.device = device
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.start_epoch = 0
        self.sampler = sampler
        self.best_valid_acc = 0
        self.lr_history = []
        self.training_time = 0
        self.start_from_checkpoint = start_from_checkpoint
        
        if optimal_threshold is None:
            self.optimal_threshold = 0.5  
        else:
            self.optimal_threshold = optimal_threshold

        self.train_loader = None
        self.test_loader = None
        self.val_loader = None

        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []

        #Intialize best val eer as infinity
        self.best_val_eer = float('inf')

        #Extract model name
        base_name = getattr(model,'architecture_name',model.__class__.__name__)

        #Set teacher present bool based on if DEIT model is active or not

        if base_name == "DEIT":
            self.model_name = f"{base_name}_{distil_type}"
            teacher_present = True
        else:
            self.model_name = base_name
            teacher_present = False

        #Set teacher model as ResNet50 and Load Resnet Model if teacher is present
        self.teacher_present = teacher_present
        if  self.teacher_present == True:
            self.teacher_resnet = self.LoadResnetModel()
            self.temperature = Temperature
            self.alpha = alpha
            self.distil_type = distil_type 

        #Set Optimizer
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.set_optimizer()

        #Initilaize a LR scheduler
        self.scheduler = CosineAnnealingLR(self.optimizer, 
                                            T_max=epochs,
                                            eta_min=1e-6)
        
        #Create save path
        save_dir = r"D:\Deep Neural Network\ML-Audio_DeepFake\Models"
        self.save_path = os.path.join(save_dir, self.model_name + ".pt")

        #Initalize the checkpoint check
        self.check_Checkpoint(save_dir)

    
        #Intialize Augmentation class for X data
        self.augmentation = Augmentation(frequency_mask_param=50,
                                         time_mask_param=30,
                                         noise_prob=0.4)

    #Load Resnet50 Model as teacher model
    def LoadResnetModel(self):
        #Load the custom Resnet 50 model
        teacher_resnet = CustomResnet50.Resnet50(num_classes=1)
        # teacher_resnet.fc = nn.Linear(teacher_resnet.fc.in_features, 1)

        #Rebuild and load the saved checkpoints from optimized Resnet50 model from the project
        checkpoint = torch.load("./Models/Resnet50.pt", map_location=self.device, weights_only=False)
        teacher_resnet.load_state_dict(checkpoint['model_state_dict'])
        teacher_resnet.to(self.device)
        teacher_resnet.eval()

        #Freeze the teacher
        for param in teacher_resnet.parameters():
            param.requires_grad = False

        return teacher_resnet

    #Set optimizer
    def set_optimizer(self):
        # if self.model_name == "CNNTRANSFORMER":
        #     self.learning_rate = 1e-4
        # else:
        #     self.learning_rate = 1e-5
            
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate,weight_decay=1e-2)
        
    
    #Set data loader for test and train
    def set_data(self, train_set, test_set, val_set):
        print(f"Number of training examples: {len(train_set)}")
        print(f"Number of testing examples: {len(test_set)}")
        print(f"Number of Val examples: {len(val_set)}")


        self.train_loader = dataloader.DataLoader(train_set, batch_size=self.batch_size,sampler=self.sampler, num_workers=0, pin_memory=True)
        self.test_loader = dataloader.DataLoader(test_set, batch_size=self.batch_size, shuffle=False, num_workers=0)
        self.val_loader = dataloader.DataLoader(val_set, batch_size=self.batch_size, shuffle=False, num_workers=0)
    
    #Compute EER 
    def compute_eer(self, labels, probabilities):
        # If the model is completely dead/flat, return 50% instead of hitting the argmin artifact
        if len(np.unique(np.round(probabilities, 5))) <= 1:
            return 50.0

        fpr, tpr, thresholds  = roc_curve(labels,probabilities,pos_label=1)
        fnr = 1-fpr

        #Find the thresholds where FPR and FNR are closest
        idx = np.nanargmin(np.absolute(fpr-fnr))
        eer = fpr[idx] * 100
        return eer
    
    
    #Checkpoint Check
    def check_Checkpoint(self,save_dir):
        #Check if the dir exits, if not then make it
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        
        if self.start_from_checkpoint:
            self.Load_checkpoint()
        else:
            #If checkpoint does exist and start_from_checkpoint is False
            #then raise an error
            #Clean the model name to safely create a filename
            model_name_clean = str(self.model_name).replace(" ", "_")
            filename = f"{model_name_clean}.pt"
            filepath = os.path.join(save_dir,filename)

            #Check if the specific file already exists
            file_exists = os.path.isfile(filepath)

            if file_exists:
                raise ValueError("Warning checkpoint exists")
            else:
                print("Starting from scratch")


    #Load checkpoint and start from best epoch
    def Load_checkpoint(self):
        #Check if checkpoint exists
        if os.path.isfile(self.save_path):
            #Load checkpoint
            checkpoint = torch.load(self.save_path, map_location=self.device, weights_only=False)

            #Checkpoint is stored as python dictionary
            #Here we unpack the dictionary to get our previous training states
            self.model.load_state_dict(checkpoint['model_state_dict'])

            self.best_valid_acc = checkpoint['best_valid_acc']

            self.train_loss = checkpoint['train_loss']
            self.train_acc = checkpoint['train_acc']
            self.val_acc = checkpoint['val_acc']
            self.val_loss = checkpoint['val_loss']

            if 'optimizer_state_dict' in checkpoint:
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    self.scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
                    self.start_epoch = checkpoint['epoch'] + 1
                    self.best_val_eer = checkpoint['best_val_eer']
                    print(f"=> Resuming full training state from epoch {self.start_epoch}")

                except (ValueError, KeyError):
                    print(" Optimizer group structure mismatch detected (Fine-Tuning LLRD active).")
                    print(" Successfully loaded pre-trained model weights! Starting optimizer fresh for fine-tuning.")
                    self.start_epoch = checkpoint['epoch'] + 1
                    # Manually anchor your target baseline EER to beat!
                    self.best_val_eer = checkpoint['best_val_eer']
                    # self.best_val_eer = 9.42

            print("Checkpoint Loaded starting from epoch:", self.start_epoch)
        else:
            raise ValueError("Chekpoint Doesn't exist")
    
    #Save Checkpoint
    def save_checkpoint(self, epoch, valid_acc):
        # self.best_valid_acc = valid_acc

        torch.save({
            "epoch" : epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            'lr_scheduler_state_dict':self.scheduler.state_dict(),
            "best_valid_acc": self.best_valid_acc,
            "best_val_eer": self.best_val_eer,
            "train_loss": self.train_loss,
            "train_acc": self.train_acc,
            "val_acc" : self.val_acc,
            "val_loss": self.val_loss
        }, self.save_path)


    #This function will perform single training epoch using our training data
    def TrainingLoop(self):

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

            X = self.augmentation.forward(X)

            #Zero out gradient
            self.optimizer.zero_grad()

            #If 
            if self.teacher_present == True:
                loss = self.Deit_Train(X,Y)
            else:
                #Forward pass
                y_pred = self.model(X)

                #Set Loss
                loss = self.loss_fn(y_pred, Y.unsqueeze(1))

            #Backpropogation and set gradient
            loss.backward()

            #Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            #Record the Learning Rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.lr_history.append(current_lr)

            #Optimization
            self.optimizer.step()

            #Keep track of train loss for plotting
            epoch_train_loss += loss.item()

        epoch_train_loss /= len(self.train_loader)

        self.train_loss.append(epoch_train_loss)
        # self.scheduler.step()

    def Deit_Train(self,X,Y):
           #Forward pass for student returns raw un activated binary logits for both heads
            logits_cls, logits_distil = self.model(X)

            #Extract raw un-activated binary logits from frozen audio native ResNet teacher
            with torch.no_grad():
                teacher_logits = self.teacher_resnet(X).squeeze(-1)
            
            #Base loss: Always evaluate the CLS head against true dataset labels
            loss_ground_truth = self.loss_fn(logits_cls,Y)
            loss_distillaton = 0

            #Distillation loss: Branch dynamically based on experiment type
            if self.distil_type == "Hard":
                #Hard Distillation
                #Calculate teacher hard labels
                teacher_hard_labels = (torch.sigmoid(teacher_logits)>=0.5).float()

                #Calcualte loss using standard BCE with logits alignment
                loss_distillaton = self.loss_fn(logits_distil,teacher_hard_labels)
            
            elif self.distil_type == "Soft":
                #Soft Distillation
                #Soften teacher's probabilty distribution using Temperature scaling
                soft_teacher_targets = torch.sigmoid(teacher_logits/self.temperature)

                #Soften student's distillation head distribution identically
                soft_student_preds = torch.sigmoid(logits_distil/self.temperature)
                
                #Calculate loss and scale gradient back by multiplying by T^2
                loss_distillaton = F.binary_cross_entropy(soft_student_preds, soft_teacher_targets)*(self.temperature**2)
            
            loss = (1-self.alpha) * loss_ground_truth + self.alpha*loss_distillaton

            return loss

        
    #Evaluation of model this runs per one epoch
    def evalModel(self, train_test_val = "test"):
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
        val_eer = 0
        sample = 0
        all_preds = []
        all_labels = []
        all_probs = []
        f1,recall = 0,0

        self.model.eval()
        with torch.inference_mode():
            for i, (X,Y) in enumerate(tqdm(loader,leave=False, desc=state)):
                #Set X, Y to device
                X,Y = X.to(self.device), Y.to(self.device)

                if self.teacher_present == True:
                    logits_cls, logits_distil = self.model(X)
                    fx = (logits_cls+logits_distil)/2
                else:   
                    #Forward pass
                    fx = self.model(X).squeeze(1) if len(self.model(X).shape) > 1 else self.model(X)

                # Ensure fx is squeezed down to [Batch] if a standard model returns [Batch, 1]
                if len(fx.shape) > 1:
                    fx = fx.squeeze(-1)

                #Loss
                loss = self.loss_fn(fx, Y)
                epoch_loss += loss.item()

                #Log sum of acc for BCE
                probs = torch.sigmoid(fx)
                probs_flat = probs.view(-1)
                preds = (probs_flat > self.optimal_threshold).float()
                labels_flat = Y.view(-1)
                
                correct_pred += (preds == labels_flat).sum().item()
                sample += labels_flat.size(0)

                #Calculate preds and labels for F1 and recall score
                all_preds.extend(preds.cpu().numpy().astype(int))
                all_labels.extend(labels_flat.cpu().numpy().astype(int))

                #Calculate probs for ROC Curve and auc score
                all_probs.extend(probs_flat.cpu().numpy())

        
        epoc_acc = correct_pred/sample
        epoch_loss /=len(loader)
        
        #Log accuracy, loss, F1 and recall from the epoch
        if train_test_val == "train":
            self.train_acc.append(epoc_acc)
            val_eer = float('inf')
            
        elif train_test_val == "val":
            self.val_acc.append(epoc_acc)
            self.val_loss.append(epoch_loss)
            f1 = f1_score(all_labels,all_preds, zero_division=0)
            recall = recall_score(all_labels,all_preds, zero_division=0)

            #Calculate EER in Val
            val_eer = self.compute_eer(all_labels,all_probs)

        elif train_test_val == "test":
            f1 = f1_score(all_labels,all_preds, zero_division=0)
            recall = recall_score(all_labels,all_preds, zero_division=0)

            #Calculate EER in test
            val_eer = self.compute_eer(all_labels,all_probs)
        
        return epoch_loss,epoc_acc,f1,recall,all_preds,all_labels,all_probs, val_eer
    
    #Run through epochs, train and evaluate the model
    def RunEpochs(self,
                  train_dataset, 
                  test_dataset, 
                  val_dataset,
                  n_epochs):
        #Set Data Loader
        self.set_data(train_set=train_dataset,test_set=test_dataset,val_set=val_dataset)

        #Train and val
        pbar = trange(self.start_epoch, n_epochs, leave=False, desc="Epoch")

        #Track of time for evaluation and learning rate history
        start_time = time.time()

        #Initializing early stopping variables
        patience = 10 #Stop training if val_loss doesn't improve for 15 epochs straight
        patience_counter = 0
        best_val_loss = float('inf')

        for epoch in pbar:
            self.TrainingLoop()

            # Step the scheduler!
            if hasattr(self, 'scheduler') and self.scheduler is not None:
                self.scheduler.step()
            
            #Calculate evaluation
            _, train_acc,_,_, _, _, _,_= self.evalModel(train_test_val="train")
            val_loss, val_acc,val_f1,val_recall, _, _, _, current_val_eer = self.evalModel(train_test_val="val")

            train_loss = self.train_loss[-1]

            print(f" Train Accuracy: {train_acc:.4f} |"
                f" Train Loss: {train_loss:.4f} |"
                f" Val Accuracy: {val_acc:.4f} |"
                f" Val Loss: {val_loss:.4f} |"
                f"F1 score: {val_f1:.4f} |"
                f"Recall score: {val_recall:.4f}")

            #Early Stopping to halt model training before overfitting
            current_val_loss = val_loss 

            if val_acc > self.best_valid_acc:
                self.best_valid_acc = val_acc

            # if current_val_eer < self.best_val_eer:
            #     self.best_val_eer = current_val_eer
            #     self.save_checkpoint(epoch,val_acc)
               

            #Check for improvement
            if current_val_eer < self.best_val_eer or epoch == self.start_epoch:
                if epoch == self.start_epoch and current_val_eer >= self.best_val_eer:
                    print(f" Initial fine-tuning epoch baseline registered at {current_val_eer:.2f}%. Saving anchor point.")
                else:
                    print(f" Validation EER improved from {self.best_val_eer:.2f}% to {current_val_eer:.2f}%!")
                    
                self.best_val_eer = current_val_eer
                patience_counter = 0 #Reset the clock because model improved

                #Save the checkpoint
                self.save_checkpoint(epoch,val_acc)

            else:
                patience_counter +=1
                print(f"Validation EER did not improve. Early Stopping counter: {patience_counter}")
            
            if patience_counter >=patience:
                print(f"Early Stopping trigger at epoch {epoch}! Stopping training to prevent overfitting")
                break
            
            
        end_time = time.time()

        self.training_time = end_time - start_time

        print(f"The highest validation accuracy was: {self.best_valid_acc:.4f}" )
        print("Training time %.2f seconds" %( self.training_time))
    

    def forward(self,x):
        return self.model(x)
        