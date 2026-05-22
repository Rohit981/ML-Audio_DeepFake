from datasets import load_dataset
import data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import os
from huggingface_hub import snapshot_download
from Resnet50 import CustomResnet50
from Trainer import Trainer
import torch.nn as nn  
from tqdm import tqdm,trange
from torchmetrics.classification import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix


os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    # snapshot_download(
    #     repo_id="RohitGENAICODER/ASVspoofLADataset",
    #     repo_type="dataset",
    #     local_dir="Datasets/LA",
    #     token="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"
    # )

    Data_root = r"D:\Deep Neural Network\ML-Audio_DeepFake\Datasets\LA"

    #Instantiate Datasets
    train_dataset = data.ASVpoofDataset(Data_root,part="train", precompute=False)
    val_dataset = data.ASVpoofDataset(Data_root, part="dev", precompute=False)
    test_dataset = data.ASVpoofDataset(Data_root, part="eval")

    # #Create a sampler for data balance
    # sampler = train_dataset.data_balancing(train_dataset)
    
    #Initialize Batch Size
    batch_size = 64
    
    #Define Loss Fn and assign weights
    train_labels = train_dataset.file_labels
    count_0 = train_labels.count(0)
    count_1 = train_labels.count(1)

    pos_weight_value = torch.tensor([count_0 / count_1], dtype=torch.float).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_value)

    #Resnet Model
    Resnet_50 = CustomResnet50.Resnet50(1)
    Resnet_18 = CustomResnet50.Resnet18(1)

    trainer = Trainer.ModelTrainer(Resnet_50, device,loss_fn,learning_rate=1e-4, batch_size=batch_size,sampler=None)

    #Set Data Loader
    trainer.set_data(train_set=train_dataset,test_set=test_dataset,val_set=val_dataset)

    #Train and val
    n_epochs = 100
    pbar = trange(trainer.start_epoch, n_epochs, leave=False, desc="Epoch")

    for epoch in pbar:
        trainer.Training()

        #Calculate evaluation
        train_loss, train_acc,_,_= trainer.eval_model(train_test_val="train")
        val_loss, val_acc,val_f1,val_recall = trainer.eval_model(train_test_val="val")

        print(f" Train Accuracy: {train_acc:.4f} |"
              f" Train Loss: {train_loss:.4f} |"
              f" Val Accuracy: {val_acc:.4f} |"
              f" Val Loss: {val_loss:.4f} |"
              f"F1 score: {val_f1:.4f} |"
              f"Recall score: {val_recall:.4f}")
        
    #Calculate Confusion Matrix
    confusion_matrix = ConfusionMatrix(task='binary')
    conf_mat = confusion_matrix(torch.tensor(trainer.all_labels), 
                                torch.tensor(trainer.all_preds))

    #Plot Confussion Matrix
    fig, ax = plot_confusion_matrix(conf_mat.numpy(),
                                    class_names=['bonafide', 'spoof'],
                                    figsize=(12,9))
    plt.show()
    

    
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
   



if __name__ == "__main__":
    main()