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
    #     local_dir="Datasets/LA",
    #     token="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"
    # )

    Data_root = r"D:\Deep Neural Network\ML-Audio_DeepFake\Datasets\LA"

    #Instantiate Datasets
    train_dataset = data.ASVpoofDataset(Data_root,part="train")
    dev_dataset = data.ASVpoofDataset(Data_root, part="dev")


    #Create a sampler for data balance
    sampler = train_dataset.data_balancing(train_dataset)

    # #Dataloader for train and dev dataset
    batch_size = 128
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

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