import os
import json
import csv
from config import AudioConfig
from datetime import datetime

class ResultLeaderboard:
    def __init__(self, config:AudioConfig):
        #Intialize dir and make it
        self.save_dir = config.leaderBoard_savedir
        os.makedirs(self.save_dir, exist_ok=True)

        #Define paths for summary tables and detailed metrics
        self.csv_path = os.path.join(self.save_dir, "Leaderboard.csv")
        self.json_path = os.path.join(self.save_dir, "detailed_metric.json")

        # #Standard tracking headers for comparitive analysis
        # self.header = [
        #     "Timestamp", "Model Name", "Highest Val Acc",
        #     "Default Test Acc", "Test ROC-AUC", "Test EER (%)", "Optimal Threshold"
        # ]

        # #Create CSV file with header is it doesn't exist
        # if not os.path.exists(self.csv_path):
        #     with open(self.csv_path, mode="w",newline="", encoding="utf-8") as f:
        #         writer = csv.writer(f)
        #         writer.writerow(self.header)
        
        #Initialize Final metric
        self.highest_val_acc = config.highest_val_acc
        self.test_acc = config.test_acc
        self.roc_auc_value = config.roc_auc_value
        self.eer_value = config.eer_value
        self.optimal_threshold = config.optimal_threshold
        
    def add_run(self,
                model_name:str,
                metrics:dict,
                classification_report:dict = None,
                confusion_matrix:list = None):
        
        #Keep tranck of the timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        #Save metrics in a CSV file
        row = [
            f"=================================== \n"
            f"Timestamp:{timestamp}",
            f"Model Name:{model_name}",
            f"Highest Val Acc:{metrics.get("Highest Val Acc",0.0):.4f}",
            f"Default Test Acc:{metrics.get("Default Test Acc",0.0):.4f}",
            f"Test ROC-AUC:{metrics.get("Test ROC-AUC",0.0):.4f}",
            f"Test EER (%):{metrics.get("Test EER (%)",0.0):.2f}",
            f"Optimal Threshold:{metrics.get("Optimal Threshold",0.0):.6f}"
            f"==================================== \n"
        ]

        #We write it to the CSV file
        with open(self.csv_path, mode='a',newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for item in row:
             writer.writerow([item])
        
        #Save metrics like confusion metric, report to JSON
        summary_metrics = {
            "timestamp":timestamp,
            "model_name":model_name,
            "summary_metrics":metrics,
            "Classification Report":classification_report,
            "Confusion Matrix": confusion_matrix
        }

        #Check for existing data and if it doesn't exist then through an error
        existing_data = {}
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, mode='r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = {}
        
        #Use the model name as the key to overwrite it with the existing model
        existing_data[model_name] = summary_metrics

        #Write it to the path
        with open(self.json_path, mode='w', encoding='utf-8') as f:
            json.dump(existing_data,f,indent=4)
        
        print(f"\n[INFO] Successfully logged results for {model_name} to {self.save_dir}/")
        self.display()
    
    def display(self):
        #Print a tabular of readout of the leaderboard
        if os.path.exists(self.csv_path):
            print("Leaderboard is empty")
            return
        
        print("\n" + "="*50)
        print(f"{'AUDIO DEEPFAKE ARCHITECTURE LEADERBOARD':^50}")
        print("="*50)

        #we read the csv file
        with open(self.csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)

            if len(rows):
                print("No runs recorded yet")
                return

            for row in rows:
                if not row:
                    continue
                line = row[0]

                if "Model Name:" in line:
                    print(f"\n {line}")
                elif any(m in line for m in ["ACC", "AUC", "EER", "Threshold"]):
                    print(f"{line}")
                else:
                    print(line)
        print("="*50 + "\n")

    #Set variable used in the main class
    def set_final_metric(self):
        final_metric = {
            "Highest Val Acc" : self.highest_val_acc,
            "Default Test Acc" : self.test_acc,
            "Test ROC-AUC" : self.roc_auc_value,
            "Test EER (%)" : self.eer_value,
            "Optimal Threshold": self.optimal_threshold
        }
        return final_metric


