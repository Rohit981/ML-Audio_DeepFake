import torch
import torch.nn as nn
from Transformer.TransformerModel import MultiHeadAttention, PositionalEncoding, PositionWiseFeedForward

class CNN(nn.Module):
    def __init__(self, 
                 input_dim=1, 
                 layer_channels=[32.64,128], 
                 dropout_value=0.2):
        super().__init__()
        #Initialize Input and Output
        self.input_dim = input_dim

        #Build the convolutional blocks
        layers = []
        in_channels = input_dim

        for out_channels in layer_channels:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.GELU(),
                    nn.Dropout(dropout_value),
                    nn.MaxPool2d(2)
                )
            )

            #Update the input pointer for the next layer block
            in_channels = out_channels
        
        #Wrap everything into a single sequence executing block
        self.feature_extractor = nn.Sequential(*layers)

        #Keep a public variable so the Transformer class knows exactly what dimension to expect
        self.final_num_channels = layer_channels[-1]

    
    def forward(self,x):
        # Pass input through the dynamically created blocks
        x = self.feature_extractor(x) # Output shape: [B, final_num_channels, H_new, W_new]

        # Flatten spatial dimensions and transpose to sequence format
        x = x.flatten(2) # [B, final_num_channels, H_new * W_new]
        x = x.transpose(1,2) # [B, H_new * W_new, final_num_channels]
        return x
    
    
#Encoder using Multiheaded attention and positional wise feed forward
class EncoderBlock(nn.Module):
    def __init__(self, 
                 d_model, 
                 num_heads,
                 d_ff, 
                 dropout_value=0.3):
        super().__init__()

        #First Layer which is Multiheaded and Layer Norm
        self.attention = MultiHeadAttention(d_model,num_heads)
        self.norm1 = nn.LayerNorm(d_model)

        #Second layer positional wise feed forward and layer norm
        self.positionalWise = PositionWiseFeedForward(d_model,d_ff)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout_value)
    
    def forward(self,x):
        #Calculate the attention output
        attn_output = self.attention(x,x,x,mask=None)

        #Adding to the original output for residual connection
        x = self.norm1(x + self.dropout(attn_output))

        #Calculate Feed Forward
        ff_output = self.positionalWise(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

#Transformer Layer uses the encoder block and does classification
class TransformerLayer(nn.Module):
    def __init__(self, 
                 d_model=128, 
                 n_heads=4, 
                 num_layers=3,
                 d_ff=512,
                 num_classes=1):
        super().__init__()

        #Positional Encoding class
        self.positional_encoding = PositionalEncoding(d_model,max_seq_length=256)
        self.dropout = nn.Dropout(0.1)

        #Sequential stack of encoder layers and d_ff = d_model X 4
        self.encoder_layers = nn.ModuleList([
            EncoderBlock(d_model,n_heads,d_ff)
            for _ in range(num_layers)
        ])

        #Classification Head outputing 1 logit for BCELogitLoss
        self.fc = nn.Linear(d_model,num_classes)
    
    def forward(self,x):
        #Input shape [B, H_new * W_new, final_num_channels]
        x = self.positional_encoding(x)
        x = self.dropout(x)

        #Loop through encoder block
        for layer in self.encoder_layers:
            x = layer(x)
        
        #Global Average Pooling across the temporal length (256 tokens)
        x = x.mean(dim=1) # Shape: [Batch, 128]

        #Final Projection layer for classification
        x = self.fc(x)
        return x

#Define a CNN Transformer Model
class CNNTrasnformer(nn.Module):
    def __init__(self, 
                 input_dim=1, 
                 layer_channels=[32,64,128], 
                 n_heads=4,
                 num_layers=3,
                 d_ff=512):
        super().__init__()

        #Initialize Dynamic CNN
        self.cnn = CNN(input_dim,layer_channels)

        #Automatically read the final number of channels
        d_model = self.cnn.final_num_channels

        #Intialize the transformer class
        self.transformer = TransformerLayer(
            d_model,
            n_heads,
            num_layers,
            d_ff,
        )
    
    def forward(self,x):
        features = self.cnn(x)
        logits = self.transformer(features)
        return logits
        


    


