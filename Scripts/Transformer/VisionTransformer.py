import torch
import torch.nn as nn
from Transformer.TransformerModel import MultiHeadAttention, PositionWiseFeedForward
from config import AudioConfig


#Patch encoding to create patches of images and will be used as positional encoder
class PatchEmbedding(nn.Module):
    def __init__(self,
                 img_size,
                 patch_size,
                 in_channels,
                 embed_dim):
        super().__init__()
        self.patch_size = patch_size

        # Convolution cuts the image into patches and projects them to embed_dim vectors
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # (128 // 16) ** 2 = 8 * 8 = 64 patches
        num_patches = (img_size // patch_size)**2

        # Learnable tokens
        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))

        # Must be num_patches + 1 to account for the CLS token, and end with embed_dim
        self.pos_embed = nn.Parameter(torch.randn(1,num_patches+1,embed_dim))
    
    def forward(self,x):
        B = x.size(0)
        x = self.proj(x) # (B,C, H/P, W/P)
        x = x.flatten(2).transpose(1,2) # (B,N,E)
        cls_token = self.cls_token.expand(B,-1,-1)
        x = torch.cat((cls_token,x),dim=1)
        x = x + self.pos_embed
        return x

#Encoder using Multiheaded attention and positional wise feed forward
class EncoderBlock(nn.Module):
    def __init__(self,
                 d_model,
                 d_ff,
                 n_heads,
                 dropout_value=0.2):
        super().__init__()

        #First layer Multiheaded attention and layer norm
        self.layernorm1 = nn.LayerNorm(d_model)
        self.multiHeaded = MultiHeadAttention(d_model,n_heads)

        #Second Layer Positional Feed forward and layer norm
        self.layernorm2 = nn.LayerNorm(d_model)
        self.positionalWise = PositionWiseFeedForward(d_model,d_ff)

        #Regularization
        self.dropout = nn.Dropout(dropout_value)
    
    def forward(self,x):

        #Calculate the attention output by using pre normalization
        norm_x_1 = self.layernorm1(x)
        attn_output = self.multiHeaded(norm_x_1,norm_x_1,norm_x_1,mask=None)
        x = x + self.dropout(attn_output) # residual connection

        #Positional Feed Forward by using pre normalization
        norm_x_2 = self.layernorm2(x)
        ff_output = self.positionalWise(norm_x_2)
        x = x + self.dropout(ff_output)
        return x

#Transformer layer uses the encoder block and does classification
class TransformerLayer(nn.Module):
    def __init__(self,
                 d_model,
                 n_heads,
                 num_layers,
                 d_ff,
                 num_classes):
        super().__init__()

        #Regularization
        self.dropout = nn.Dropout(0.2)

        #Sequential stack of encoder layer
        self.encoder_layer = nn.ModuleList([
            EncoderBlock(d_model,d_ff,n_heads)
            for _ in range(num_layers)
        ])

        #Classification Head outputing 1 logits for BCELogitsLoss
        self.fc = nn.Linear(d_model,num_classes)
    
    def forward(self,x):
        x = self.dropout(x)

        #loop through encoder block
        for layer in self.encoder_layer:
            x = layer(x)
        
        # Standard ViT Method: Slice out just the CLS Token vector
        x = x[:,0] # Shape: [Batch, d_model]

        #Final Projection layer
        x = self.fc(x)
        return x

#Vision Transformer
class VIT(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()

        #Convert image space to token sequence space
        self.patchembedding = PatchEmbedding(
            img_size=config.img_size,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=config.d_model
        )

        #Process token using multi headed self attention
        self.transformerLayer = TransformerLayer(
            d_model=config.d_model,
            n_heads=config.n_heads,
            num_layers=config.num_layers,
            d_ff=config.d_ff,
            num_classes=config.num_classes
        )
    
    def forward(self,x):
        # Input x shape: [Batch, 1, 128, 128]
        
        # Run through patch embedding
        # Output shape: [Batch, 65, d_model]
        x = self.patchembedding(x)

        #Pass token vectors directly into the encoder blocks & classification head
        # Output shape: [Batch, 1]
        x = self.transformerLayer(x)

        return x






