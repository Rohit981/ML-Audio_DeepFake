from dataclasses import dataclass,field
import torch
import random
import numpy as np


@dataclass
class AudioConfig:
     #------------------Hardware selection ------------------------------
     device : torch.device = "cuda" if torch.cuda.is_available() else "cpu"

     #----------------------Save Directory-------------------------------
     leaderBoard_savedir : str = r"D:\Deep Neural Network\ML-Audio_DeepFake\Evaluation\Metric\LeaderBoard"
     classification_savedir : str = r"D:\Deep Neural Network\ML-Audio_DeepFake\Evaluation\Metric\Confusion&Classification Report"

     #----------------------LeaderBoard Metrics--------------------------
     highest_val_acc: float = None
     test_acc: float = None
     roc_auc_value: float = None
     eer_value: float = None
     optimal_threshold: float = None
     #------------------Data / Spectogram spec ----------------------------
     img_size : int = 128    #Height and width of mel spec
     patch_size : int = 16   #Base patch size for VIT and DEIT
     in_channels : int = 1   #Grayscale audio spectogram
     num_classes : int = 1   #Binary classification

     #---------------------Core Architecture Parameters----------------------
     d_model : int = 128    #Baseline dimension of hidden state
     n_heads : int = 4      #Number of heads used by transformer
     num_layers: int = 3    #Layer configration
     d_ff : int = d_model*4
     input_layer_channels : list = field(default_factory = lambda:[32,64,128])

     #--------------------Training and Optimization-------------------------
     batch_size: int = 64
     learning_rate: float = 2e-5
     n_epochs : int = 500
     start_from_checkpoint : bool = True
     Temperature: float = 1.0 #Only use if distill is soft
     alpha: float = 0.5 #50/50 balance between ground truth and distillation
     distil_type: str = "Hard" #Option hard or soft

     def set_seed(self,seed=42):
          random.seed(seed)
          np.random.seed(seed)
          torch.manual_seed(seed)
          torch.cuda.manual_seed(seed)
          torch.cuda.manual_seed_all(seed) # if using multi-GPU
          torch.backends.cudnn.deterministic = True
          torch.backends.cudnn.benchmark = False

     
