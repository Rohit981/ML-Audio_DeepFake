from __future__ import annotations # Allows deferred evaluation of type hints
import torch
from typing import TYPE_CHECKING
from config import AudioConfig

if TYPE_CHECKING:
    from Trainer import ModelTrainer


def get_swin_optimizer(model, base_lr = 5e-6, decay_rate=1.0):
    parameters_group = []

    #Patch embedding and early stages
    parameters_group.append({
        "params" : model.patch_embed.parameters(),
        "lr" : base_lr * (decay_rate ** 2) # 4e-6
    })

   #Gathering parameters from stage1, patch_merge, stage2 and pre-head normalization
    core_transformer_params = (
        list(model.stage1_block.parameters()) +
        list(model.patch_merge.parameters()) +
        list(model.stage2_block.parameters()) +
        list(model.norm.parameters())
    )

    parameters_group.append({
        "params" : core_transformer_params,
        "lr" : base_lr*decay_rate
    })
            
    # Final Classification Head
    parameters_group.append({
        "params": model.fc.parameters(), 
        "lr": base_lr # e.g., 1e-5
    })
    
    optimizer = torch.optim.AdamW(parameters_group, weight_decay=1.0)
    return optimizer

def get_vit_or_deit_optimizer(model, base_lr=2e-5, decay_rate=1.0):
    parameters_group = []
    
    # 1. Patch Embeddings & Positional Embeddings (Lowest LR)
    embed_params = [p for n, p in model.named_parameters() if "patch_embed" in n or "pos_embed" in n or "cls_token" in n]
    parameters_group.append({
        "params": embed_params,
        "lr": base_lr * (decay_rate ** 2)
    })
    
    # 2. Core Transformer Blocks (Intermediate LR)
    # Note: For strict LLRD, you can decay layer-by-layer, but grouping all blocks works too.
    body_params = [p for n, p in model.named_parameters() if "blocks" in n or "norm" in n]
    parameters_group.append({
        "params": body_params,
        "lr": base_lr * decay_rate
    })
    
    # 3. Head (Highest LR)
    head_params = [p for n, p in model.named_parameters() if "head" in n or "fc" in n]
    parameters_group.append({
        "params": head_params,
        "lr": base_lr
    })
    
    return torch.optim.AdamW(parameters_group, weight_decay=1.0)

def get_generic_finetune_optimizer(model, base_lr=5e-2, decay_rate=1.0):
    """
    Dynamically groups parameters into three tiers:
    Tier 1 (Lowest LR): Early feature extraction (CNN stems, patch embeddings)
    Tier 2 (Mid LR): Core sequential processing blocks (CNN stages, Transformers blocks)
    Tier 3 (Base LR): Prediction head
    """
    tier1_params = []
    tier2_params = []
    tier3_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Target the final classification layers
        if "fc" in name or "head" in name or "classifier" in name:
            tier3_params.append(param)
            
        # Target early downsampling stems (e.g., 'cnn_stem', 'patch_embed', 'conv1')
        elif "patch_embed" in name or "stem" in name or "conv1" in name or "pos_embed" in name:
            tier1_params.append(param)
            
        # Everything else falls into the core block category (stages, blocks, bottlenecks)
        else:
            tier2_params.append(param)
            
    parameters_group = [
        {"params": tier1_params, "lr": base_lr * (decay_rate ** 2)},
        {"params": tier2_params, "lr": base_lr * decay_rate},
        {"params": tier3_params, "lr": base_lr}
    ]
    
    return torch.optim.AdamW(parameters_group, weight_decay=0.08)

def set_Optimizers(model_name, 
                   active_model,
                   trainer: ModelTrainer,
                   config : AudioConfig):
    #Set Fine Tune Optimizer for Swin
    if model_name == "SwinTransformer":
        print("SWIN FINE TUNE OPTIMIZER")
        fine_tune_optimizer = get_swin_optimizer(active_model)
    elif model_name == "VIT" or model_name == "DEIT":
        print("VIT OR DEIT FINE TUNE OPTIMIZER")
        fine_tune_optimizer = get_vit_or_deit_optimizer(active_model)
    else:
        print("Other Models")
        fine_tune_optimizer = get_generic_finetune_optimizer(active_model)
    
    #Explicity override the default optimizer inside the trainer instance
    trainer.optimizer = fine_tune_optimizer

    #Warup Setup Configuration
    warmup_epochs = getattr(config, 'warmup_epochs', 2)

    if warmup_epochs > 0:
        print(f"Setting up {warmup_epochs} Warmup Epochs followed by Cosine Annealing")

        # 1. Warmup Scheduler: Linearly scale from 10% (0.1) to 100% (1.0) of maximum learning rate
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            trainer.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs
        )

        #Main Decay Scheduler: Run for the remaining budget of epochs
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            trainer.optimizer,
            T_max=(config.n_epochs - warmup_epochs),
            eta_min=1e-7
        )

        #Combine them sequentially
        #milestones=[warmup_epochs] means it switches to main_scheduler at that exact epoch index
        trainer.scheduler = torch.optim.lr_scheduler.SequentialLR(
            trainer.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs]
        )
    else:
        print("No warmup phase. Proceeding with pure Cosine Annealing")
        trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            trainer.optimizer,
            T_max=config.n_epochs,
            eta_min=1e-7
        )

    
    

   
