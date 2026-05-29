import data
import torch
import os
from Resnet50 import CustomResnet50
from Trainer import Trainer
import torch.nn as nn  
from Transformer import CnnTrasnformer,VisionTransformer
import evaluation
from config import AudioConfig
from Utils.Leaderboard import ResultLeaderboard


os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

def main():
    
    #Initialize Config class
    config = AudioConfig()


    #Instantiate Datasets
    train_dataset = data.ASVpoofDataset(part="train", precompute=False)
    val_dataset = data.ASVpoofDataset(part="dev", precompute=False)
    test_dataset = data.ASVpoofDataset(part="eval",precompute=False)

    # #Create a sampler for data balance
    sampler = train_dataset.data_balancing(train_dataset)
    
    loss_fn = nn.BCEWithLogitsLoss()

    #Resnet Models
    Resnet_50 = CustomResnet50.Resnet50(1)
    Resnet_18 = CustomResnet50.Resnet18(1)

    #CNN Transformer Model
    CNN_Transformer = CnnTrasnformer.CNNTrasnformer(config)

    #Vision Transformer
    VIT = VisionTransformer.VIT(config)

    #Track of active model
    active_model = Resnet_50

    #Initialize Trainer and run the epochs
    trainer = Trainer.ModelTrainer(model=active_model, 
                                   device=config.device,
                                   loss_fn=loss_fn,
                                   learning_rate=config.learning_rate, 
                                   batch_size=config.batch_size,
                                   sampler=sampler,
                                   start_from_checkpoint=config.start_from_checkpoint)

    trainer.RunEpochs(train_dataset=train_dataset,
                      test_dataset=test_dataset,
                      val_dataset=val_dataset,
                      n_epochs=config.n_epochs)
    
    #Initialize Evaluation Metric and set best valid acc
    config.highest_val_acc = trainer.best_valid_acc
    eval_metric =  evaluation.Evaluation_metric(Trainer=trainer,
                                 model=active_model,
                                 total_training_time=trainer.training_time,
                                 config=config)
    
    #Initialize Leaderboard and set config variables
    config.test_acc = eval_metric.test_acc
    config.roc_auc_value = eval_metric.roc_value
    config.eer_value = eval_metric.eer_value
    config.optimal_threshold = eval_metric.optimal_threshold


    leaderboard = ResultLeaderboard(config=config)
    leaderboard.add_run(
        model_name=trainer.model_name,
        metrics=leaderboard.set_final_metric(),
        classification_report=eval_metric.classification_report,
        confusion_matrix=eval_metric.Conf_list

    )
    

if __name__ == "__main__":
    main()