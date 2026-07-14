from torchmetrics.classification import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_curve,classification_report,roc_auc_score
import numpy as np
import os
import csv
from config import AudioConfig
import json
from adjustText import adjust_text


class Evaluation_metric:
    def __init__(self,
                 Trainer, 
                 model, 
                 total_training_time,
                 config:AudioConfig,
                 distil_type):
        
        #Initialize Variables
        self.trainer = Trainer
        #Extract model name
        base_name = getattr(model,'architecture_name',model.__class__.__name__)

        #Set DEIT model is active or not
        if base_name == "DEIT":
            self.model_name = f"{base_name}_{distil_type}"
        else:
            self.model_name = base_name
            
        self.n_epochs = config.n_epochs
        self.total_training_time = total_training_time
        self.highest_val_acc = config.highest_val_acc
        self.roc_value = config.roc_auc_value
        self.eer_value = config.eer_value
        self.optimal_threshold = config.optimal_threshold
        self.test_acc = config.test_acc
        self.batch_size = config.batch_size
        self.learning_rate = config.learning_rate
        self.metric_save_dir = r"D:\Deep Neural Network\ML-Audio_DeepFake\Evaluation\Metric"

        self.forward()
    
    #Log model Metric to write to the file
    def log_model_metric(self, save_dir, metrics_dict):
        #Dynamically create or appends to a CSV file named after the specific model
        os.makedirs(save_dir,exist_ok=True)

        #Clean the model name to safely create a filename
        model_name_clean = str(metrics_dict['Model Name']).replace(" ", "_")
        filename = f"{model_name_clean}_metrics.csv"
        filepath = os.path.join(save_dir,filename)

        #Check if the specific file already exists
        file_exists = os.path.isfile(filepath)

        #Define standardized columns for the fields
        fieldname = [
            "Model Name",  "Epochs", "Training Time",
            "Highest Val Acc", "Default Test Acc","Test ROC-AUC",
             "Test EER (%)", "Optimal Threshold"
        ]

        with open(filepath,mode='a', encoding='utf-8') as f:
            writer = csv.writer(f)
            # If the file already has a model log, add a visual separator before appending the new run
            if file_exists:
                f.write("\n" + "="*40 + "\n")
                f.write("NEXT TRIAL RUN STATISTICS:\n")
                f.write("="*40 + "\n")
            else:
                f.write("="*40 + "\n")
                f.write(f"INITIAL EVALUATION: {metrics_dict['Model Name']}\n")
                f.write("="*40 + "\n")
                
            # 3. Write each metric explicitly on a brand new row
            f.write(f"Model Name: {metrics_dict['Model Name']}\n")
            f.write(f"Epochs: {metrics_dict['Epochs']}\n")
            f.write(f"Training Time: {metrics_dict['Training Time']:.2f} seconds\n")
            f.write(f"Highest Val Acc: {metrics_dict['Highest Val Acc']:.4f}\n")
            f.write(f"Default Test Acc: {metrics_dict['Default Test Acc']:.4f}\n")
            f.write(f"Test ROC-AUC: {metrics_dict['Test ROC-AUC']:.4f}\n")
            f.write(f"Test EER (%): {metrics_dict['Test EER (%)']:.2f}%\n")
            f.write(f"Optimal Threshold: {metrics_dict['Optimal Threshold']:.6f}\n")
            f.write(f"Batch Size: {metrics_dict['Batch Size']}\n")
            f.write(f"Learning Rate: {metrics_dict['Learning Rate']}\n")


        print(f"Successfully logged metrics to {filepath}")
    
    #Save metrics in CSV File
    def save_metrics(self,test_acc, roc_score,eer_value, opt_threshold):
        #Package all performance details into a dictionary for CSV headers
        metrics = {
            "Model Name": self.model_name,
            "Epochs": self.n_epochs,
            "Training Time": self.total_training_time,
            "Highest Val Acc": round(self.highest_val_acc,4),
            "Default Test Acc": round(test_acc,4),
            "Test ROC-AUC": round(roc_score,4),
            "Test EER (%)": round(eer_value,2),
            "Optimal Threshold": round(opt_threshold,6),
            "Batch Size": int(self.batch_size),
            "Learning Rate": float(self.learning_rate)
        }

        #Save the File
        self.log_model_metric(self.metric_save_dir,metrics)

        #Plot Accuracy vs Speed graph
        json_path = self.save_leaderboard_metric(eer_value, test_acc,roc_score)
        self.Plot_Speed_Accuracy(json_path)
    
    #Save Leaderboard Metric
    def save_leaderboard_metric(self, eer_value, test_acc, roc_value):
        #Define json path and model name variable
        json_path = "leaderboard_PreFineTuning.json"
        model_name = self.model_name

        #Read existing JSON database if it exist
        leaderboard_data = {}
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                try:
                    leaderboard_data = json.load(f)
                except json.JSONDecodeError:
                    pass
        
        #Check if we should update(if model is new, or if current EER is better/equal)
        # if eer_value > 0:
        #     should_update = True
        #     if model_name in leaderboard_data:
        #         older_eer = leaderboard_data[model_name].get("Test EER (%)", 100.0)

        #         if eer_value > older_eer:
        #             should_update = False

        #     if should_update:
        #         leaderboard_data[model_name] = {
        #         "Highest Val Acc": float(self.highest_val_acc),
        #         "Test Acc": float(test_acc),  #calibrated accuracy
        #         "Test ROC-AUC": float(roc_value),
        #         "Test EER (%)": float(eer_value),
        #         "Training Time (s)": float(self.total_training_time) # Captured from timer
        #         }
            

        #     # Write back out to the master ledger file
        #     with open(json_path, "w", encoding="utf-8") as f:
        #         json.dump(leaderboard_data, f, indent=4)

        return json_path

    #Plot Save directory
    def plot_save_dir(self,printstr=None, save_str=None):
         #Check if the dir exits, if not then make it
        plot_dir = r"D:\Deep Neural Network\ML-Audio_DeepFake\Evaluation\Graph"
        os.makedirs(plot_dir,exist_ok=True)

        if plot_dir:
            save_path = os.path.join(plot_dir, f"{self.model_name}_{save_str}")
            plt.savefig(save_path,bbox_inches='tight',dpi=300)
            print(f"{printstr}{save_path}")
        plt.close()
    
    #Visualize Train and Test loss and Accuracy
    def Visualize_loss_acc(self,train_loss, train_acc, val_loss, val_acc, lr_history=[]):
        plt.figure(figsize=(12,5))
        epochs = range(1, len(train_loss) + 1)

        #Loss History
        plt.subplot(1,3,1)
        plt.plot(epochs,train_loss,label="Train Loss", color="blue")
        plt.plot(epochs,val_loss,label="Val_loss", color="red")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss History")
        plt.legend()
        plt.grid(True)

        #Accuracy History
        plt.subplot(1,3,2)
        plt.plot(epochs,train_acc,label="Train Accuracy", color="blue")
        plt.plot(epochs,val_acc,label="Val Accuracy",color="red")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy History")
        plt.legend()
        plt.grid(True)

        #Plot Learning Rate vs Epoch
        steps = range(len(lr_history))
        plt.subplot(1,3,3)
        plt.plot(steps,lr_history, label="LR")
        plt.xlabel("Steps")
        plt.ylabel("Learning Rate")
        plt.title("Learning Scheduler")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
       
        self.plot_save_dir(printstr="Training history graphs saved to", 
                           save_str="training_history")

        

        #Visualize Augmentation
        # original_img,_ = dev_dataset[0]
        # augmented_img,_ = train_dataset[0]

        # plt.figure(figsize=(12,5))

        # plt.subplot(1,3,1)
        # plt.imshow(
        #     original_img.squeeze().numpy(),
        #     aspect='auto',
        #     origin='lower',
        #     cmap='magma'
        #     )
        # plt.title('Original Image')

        # plt.subplot(1,3,2)
        # plt.imshow(
        #     augmented_img.squeeze().numpy(),
        #     aspect='auto',
        #     origin='lower',
        #     cmap='magma'
        #     )
        # plt.title('Augmented Image')

        # plt.subplot(1,3,3)
        # difference = augmented_img - original_img
        # plt.imshow(
        #     difference.squeeze().numpy(),
        #     aspect='auto',
        #     origin='lower',
        #     cmap='magma'
        # )
        # plt.title("Injected Noise")

        # plt.tight_layout()
        # plt.show()

    #Plot Speed vs Accuracy graph
    def Plot_Speed_Accuracy(self,
                            json_path = "leaderboard_PreFineTuning.json"):
        
        #Check if the file exist or not
        if not os.path.exists(json_path):
            print(f"{json_path} not found. Skipping plot generation until data is saved.")
            return
        
        #Read the master JSON dataset
        with open(json_path, mode="r", encoding="utf-8") as f:
            leaderboard_data = json.load(f)
        
        #Set up the plot enviornment
        plt.figure(figsize=(12,5))
        plt.grid(True,which="both",ls="--", alpha=0.5)

    
        #Loop through every model present in the JSON file dynamically
        for model_name, metrics in leaderboard_data.items():
            if model_name == "DEIT_hard" or "DEIT_Hard" in model_name.upper() and "_" not in model_name:
                continue

            #Pull coordinates from the dict
            training_time = metrics.get("Training Time (s)", 0.0)
            test_acc = metrics.get("Test Acc",0.0)*100 #Convert to percentage
            

            #Don't plot if the metrics are not recorded
            if training_time == 0 or test_acc == 0:
                continue
            
            # Choose distinctive publication markers and colors based on architecture type
            if "DEIT" in model_name.upper():
                color, marker = 'red', 'D'        # Distilled Student variants
            elif "CNNTRANSFORMER" in model_name.upper():
                color, marker = 'darkred', '*'    # Hybrid model
            elif "RESNET" in model_name.upper():
                color, marker = 'forestgreen', 's' # CNN baselines
            else:
                color, marker = 'dodgerblue', 'o'  # SWIN, VIT baselines

            #Plot Point
            plt.scatter(training_time, test_acc, color=color,marker=marker, s=100, zorder=5)

            plt.annotate(f" {model_name}", (training_time, test_acc), 
                         fontsize=9, 
                         fontweight='bold' if "CNN" in model_name.upper() else 'normal',
                         va='center', 
                         ha='left')
            
        #Graph customizations (lower training time means a faster model)
        plt.xlabel("Total Training Time (seconds)", fontsize=11, fontweight='bold')
        plt.ylabel("Calibrated Test Accuracy (%)", fontsize=11, fontweight='bold')
        plt.title("Performance vs. Training Efficiency Frontier", fontsize=12, fontweight='bold', pad=15)
        
        
        plt.tight_layout()
        save_path = r'D:\Deep Neural Network\ML-Audio_DeepFake\Evaluation\Graph\Speed_Vs_Accuracy_Frontier_PreFineTuning.svg'
        plt.savefig(
            save_path, 
            dpi=300, 
            bbox_inches='tight', 
            transparent=True
        )
        plt.close()
        print("Dynamic 'speed_vs_accuracy_frontier.png' successfully refreshed from JSON file!")
   
    #Calculate the ROC and AUC values
    def ROC_AUC_Values(self,y_true,y_probs):
        #Calculate the ROC curve coordinates
        fpr, tpr, thresholds = roc_curve(y_true,y_probs,pos_label=1)
        fnr = 1 - tpr

        #Calculate ROC AUC Score
        auc_score = roc_auc_score(y_true,y_probs)
        print(f"ROC AUC Score:{auc_score:.4f}")

        #Locate the intersection point where FPR ~= FNR
        idx = np.nanargmin(np.absolute(fpr - fnr))
        eer = fpr[idx]
        self.optimal_threshold = thresholds[idx]

        print("\n" + "="*40)
        print(f" Equal Error Rate (EER): {eer * 100:.2f}%")
        print(f" Optimal Decision Threshold: {self.optimal_threshold:.6f}")
        print("="*40 + "\n")
        
        #Apply the optimal threshold to map new predictions
        tuned_preds = [1 if p > self.optimal_threshold else 0 for p in y_probs]
        
        #Generate the adjusted report
        print("Tuned Classification Report:")
        self.classification_report = classification_report(y_true, 
                                                           tuned_preds, 
                                                           target_names=['bonafide', 'spoof'],
                                                           output_dict=True)
        print(self.classification_report)

        #Plot the ROC curve
        plt.figure(figsize=(6,6))
        plt.plot(fpr,tpr, label=f'AUC = {auc_score:.4f}')
        plt.plot([0,1], [0,1], "--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid(True)

        self.plot_save_dir(printstr="ROC Curve saved to",
                           save_str="roc_curve")


        return auc_score, eer*100, self.optimal_threshold
    
    def forward(self):
        #Call the evaluate function and pass the evaluation/test dataloader
        test_loss, self.test_acc, test_f1, test_recall, test_preds, test_labels, test_probs,_ = self.trainer.evalModel(train_test_val="test")
        print(f"Raw Default Accuracy: {self.test_acc:.4f}")

        #Plot Graph to visualize Train loss, acc and val loss, acc
        self.Visualize_loss_acc(self.trainer.train_loss,
                                self.trainer.train_acc,
                                self.trainer.val_loss,
                                self.trainer.val_acc,
                                self.trainer.lr_history)
        
        #Plot ROC curve,AUC score and classification Report
        self.roc_value, self.eer_value, self.optimal_threshold = self.ROC_AUC_Values(test_labels,test_probs)

        #Override self.test_acc with the true tuned accuracy!
        self.test_acc = self.classification_report['accuracy']
        print(f"Calibrated Test Accuracy: {self.test_acc:.4f}")

        # Generate the optimized Confusion Matrix using the tuned threshold
        tuned_preds_tensor = torch.tensor([1 if p > self.optimal_threshold else 0 for p in test_probs])
            
        #Calculate Confusion Matrix
        confusion_matrix = ConfusionMatrix(task='binary')
        conf_mat = confusion_matrix(torch.tensor(test_labels), 
                                    tuned_preds_tensor)
        
        #Create a metric list that will be used by leaderboard
        self.Conf_list = conf_mat.tolist()
        
        #Plot Confussion Matrix
        fig, ax = plot_confusion_matrix(conf_mat.numpy(),
                                        class_names=['bonafide', 'spoof'],
                                        figsize=(12,9))
        plt.show()
        
        #Save the metrics
        self.save_metrics(self.test_acc,self.roc_value,self.eer_value,self.optimal_threshold)

       
    