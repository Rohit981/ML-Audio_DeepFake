import data
import torch
import os
from Resnet50 import CustomResnet50
from Trainer import Trainer
import torch.nn as nn  
from Transformer import CnnTrasnformer,VisionTransformer
import evaluation


os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    #Instantiate Datasets
    train_dataset = data.ASVpoofDataset(part="train", precompute=False)
    val_dataset = data.ASVpoofDataset(part="dev", precompute=False)
    test_dataset = data.ASVpoofDataset(part="eval",precompute=False)

    # #Create a sampler for data balance
    sampler = train_dataset.data_balancing(train_dataset)
    
    #Initialize Batch Size, epochs and checkpoints
    batch_size = 64
    n_epochs = 100
    start_from_checkpoint = False
    
    #Define Loss Fn and assign weights
    # train_labels = train_dataset.file_labels
    # count_0 = train_labels.count(0)
    # count_1 = train_labels.count(1)

    # pos_weight_value = torch.tensor([count_0 / count_1], dtype=torch.float).to(device)
    loss_fn = nn.BCEWithLogitsLoss()

    #Resnet Models
    Resnet_50 = CustomResnet50.Resnet50(1)
    Resnet_18 = CustomResnet50.Resnet18(1)

    #CNN Transformer Model
    CNN_Transformer = CnnTrasnformer.CNNTrasnformer()

    #Vision Transformer
    VIT = VisionTransformer.VIT()

    #Track of active model
    active_model = VIT

    #Initialize Trainer and run the epochs
    trainer = Trainer.ModelTrainer(model=active_model, 
                                   device=device,
                                   loss_fn=loss_fn,
                                   learning_rate=1e-4, 
                                   batch_size=batch_size,
                                   sampler=sampler,
                                   start_from_checkpoint=start_from_checkpoint)

    trainer.RunEpochs(train_dataset=train_dataset,
                      test_dataset=test_dataset,
                      val_dataset=val_dataset,
                      n_epochs=n_epochs)

    evaluation.Evaluation_metric(Trainer=trainer,
                                 model=active_model,
                                 n_epochs=n_epochs,
                                 total_training_time=trainer.training_time,
                                 highest_val_acc=trainer.best_valid_acc)

           

if __name__ == "__main__":
    main()