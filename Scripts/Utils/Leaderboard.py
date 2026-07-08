import os
import json
import csv
from config import AudioConfig
from datetime import datetime

class ResultLeaderboard:
    def __init__(self, 
                 config:AudioConfig,
                 model_name: str):
        #Intialize dir and make it
        self.leaderboard_dir = config.leaderBoard_savedir
        self.classification_dir = config.classification_savedir
        self.model_name = model_name
        os.makedirs(self.leaderboard_dir, exist_ok=True)
        os.makedirs(self.classification_dir, exist_ok=True)

        #Define paths for summary tables and detailed metrics
        self.csv_path = os.path.join(self.leaderboard_dir, "Leaderboard_AfterFineTuned.csv")
        self.json_path = os.path.join(self.classification_dir, f"{model_name}_Confusion&Classification_Report.json")


        #Initialize Final metric
        self.highest_val_acc = config.highest_val_acc
        self.test_acc = config.test_acc
        self.roc_auc_value = config.roc_auc_value
        self.eer_value = config.eer_value
        self.optimal_threshold = config.optimal_threshold
        
    def add_run(self,
                metrics:dict,
                classification_report:dict = None,
                confusion_matrix:list = None):
        
        #Keep tranck of the timestamp and set model name
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        #Set the Leaderboard
        self.Set_Leaderboard(timestamp=timestamp,
                             model_name=self.model_name,
                             metrics=metrics)
        
        #Set the Classification and Confusion matrix
        self.Set_SummaryReport(timestamp=timestamp,
                               model_name=self.model_name,
                               classification_report=classification_report,
                               confusion_matrix=confusion_matrix)

        self.display()
    
    def Set_Leaderboard(self,
                        timestamp:str,
                        model_name:str,
                        metrics:dict):
        # Fetch the test EER safely as a float first
        try:
            current_eer = float(metrics.get('Test EER (%)', 0.0))
        except (ValueError, TypeError):
            current_eer = 0.0

        # CRITICAL FILTER: Ignore this run completely if EER is invalid or <= 0.0%
        if current_eer <= 0.0:
            print(f"Skipping leaderboard update for {model_name}: Invalid or 0.0% EER detected.")
            return

        # Build current run with safe formatting
        current_run = {
            "Time Stamp": timestamp,
            "Model Name": model_name,
            "Highest Val Acc": f"{metrics.get('Highest Val Acc', 0.0):.4f}",
            "Default Test Acc": f"{metrics.get('Default Test Acc', 0.0):.4f}",
            "Test ROC-AUC": f"{metrics.get('Test ROC-AUC', 0.0):.4f}",
            "Test EER (%)": f"{current_eer:.2f}%"
        }

        all_runs = [current_run]
        
        # Read and parse previous entries out of the leaderboard file
        if os.path.exists(self.csv_path):
            with open(self.csv_path, mode='r', newline="", encoding="utf-8") as f:
                lines = f.readlines()
            
            for line in lines:
                if "|" in line and "Model Name" not in line and "----" not in line and "====" not in line and "ARCHITECTURE" not in line:
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) >= 5:
                        try:
                            # Validate historical EER before adding it back
                            hist_eer_val = float(parts[4].replace("%", "").strip())
                            if hist_eer_val <= 0.0:
                                continue  # Clean out any old corrupted 0.0 entries
                        except ValueError:
                            continue

                        all_runs.append({
                            "Time Stamp": "Historical", # Maintain structure symmetry
                            "Model Name": parts[0],
                            "Highest Val Acc": parts[1],
                            "Default Test Acc": parts[2],
                            "Test ROC-AUC": parts[3],
                            "Test EER (%)": parts[4]
                        })
        
        # Deduplicate historical data runs (keep the absolute lowest non-zero EER)
        unique_runs = {}
        for r in all_runs:
            name = r["Model Name"]
            try:
                eer_val = float(r["Test EER (%)"].replace("%", "").strip())
            except ValueError:
                continue

            if name not in unique_runs:
                unique_runs[name] = r
            else:
                try:
                    older_eer = float(unique_runs[name]["Test EER (%)"].replace("%", "").strip())
                    # Check if the new run is strictly better
                    if eer_val < older_eer:
                        unique_runs[name] = r
                except ValueError:
                    pass

        sorted_runs = list(unique_runs.values())

        # Sort entries dynamically by the lowest valid EER value
        def get_sort_key(x):
            try:
                return float(x["Test EER (%)"].replace('%', '').strip())
            except ValueError:
                return float('inf') # Push bad entries to the bottom if parsing fails

        sorted_runs.sort(key=get_sort_key)

        # Generate layout string format
        table_output = []
        table_output.append(f"{'AUDIO DEEPFAKE ARCHITECTURE LEADERBOARD (RANKED BY BEST EER)':^85}\n")
        table_output.append("="*85 + "\n")
        table_output.append(f"{'Model Name':<20} | {'Highest Val Acc':<15} | {'Test Acc':<10} | {'Test ROC-AUC':<12} | {'Test EER (%)':<12}\n")
        table_output.append("-"*85 + "\n")

        for run in sorted_runs:
            table_output.append(
                f"{run['Model Name']:<20} | "
                f"{run['Highest Val Acc']:<15} | "
                f"{run['Default Test Acc']:<10} | "
                f"{run['Test ROC-AUC']:<12} | "
                f"{run['Test EER (%)']:<12}\n"
            )

        # Overwrite clean data to file
        with open(self.csv_path, mode='w', encoding="utf-8") as f:
            f.writelines(table_output)
        
    def Set_SummaryReport(self,
                          timestamp:str,
                          model_name:str,
                          classification_report:dict,
                          confusion_matrix: list):
        
        # #Save metrics like confusion metric, report to JSON
        summary_metrics = {
            "timestamp":timestamp,
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
        
        print(f"\n[INFO] Successfully logged results for {model_name} to {self.classification_dir}/")

        
    def display(self):
        #Print a tabular of readout of the leaderboard
        if not os.path.exists(self.csv_path):
            print("Leaderboard file does not exist yet")
            return
        
        #Output the exact text file formatting directly to the terminal console
        with open(self.csv_path, mode='r', encoding='utf-8') as f:
            print("\n" + f.read())
        
                    

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


