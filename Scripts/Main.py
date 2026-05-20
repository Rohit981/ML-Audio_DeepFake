from datasets import load_dataset
import data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import os
from huggingface_hub import snapshot_download

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    # snapshot_download(
    #     repo_id="RohitGENAICODER/ASVspoofLADataset",
    #     repo_type="dataset",
    #     local_dir="Datasets/LA"
    # )

    Data_root = r"D:\Deep Neural Network\ML-Audio_DeepFake\Datasets\LA"

    #Instantiate Datasets
    train_dataset = data.ASVpoofDataset(Data_root,part="train")
    dev_dataset = data.ASVpoofDataset(Data_root, part="dev")

    #Verify Train set labels
    count_0 = train_dataset.file_labels.count(0)
    count_1 = train_dataset.file_labels.count(1)
    print(f"Count_0: {count_0}, Count_1: {count_1}")


    # #Dataloader for train and dev dataset
    # batch_size = 128
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    # #Testing out labels
    # count_0 = 0
    # count_1 = 0
    # for i in range(1000):
    #     _,label = train_dataloader.dataset[i]

    #     if label.item() == 0:
    #         count_0 +=1
    #     else:
    #         count_1 += 1
        
    #     print(f"Count 0: {count_0}, Count 1 : {count_1}")

    # X,y = next(iter(train_dataloader))
    # print(X.shape)
    # print(y.shape)

if __name__ == "__main__":
    main()