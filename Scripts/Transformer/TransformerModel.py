import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

#Multi headed class
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention,self).__init__()
        #Ensure that the model dimension (model) is divisible by number of heads
        assert d_model % num_heads == 0,"Model must be divisible by number of heads"

        #Initialize variabels
        self.d_model = d_model #Model Dimensionality
        self.num_heads = num_heads #Number of attention heads
        self.d_k = d_model // num_heads #Dimensions of each heads query, key, values

        #Linear layers for transforming inputs
        self.w_q = nn.Linear(d_model,d_model) #Query Transformer
        self.w_k = nn.Linear(d_model, d_model) # Key Transformer
        self.w_v = nn.Linear(d_model,d_model) # Value Transformer
        self.w_o = nn.Linear(d_model,d_model) # Output Transformer

    def scaled_dot_product_attention(self,Q,K,V, mask=None):
        #Calculate the attention score by calculating the dot product and dividing it by dimension of each heads
        attention_score = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d_k)

        #Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            attention_score = attention_score.masked_fill(mask== 0, -1e9)
        
        #Apply softmax to calculate attention probability or attention weights
        attention_prob = torch.softmax(attention_score,dim=-1)

        #Calculat the dot product of attention prob and v
        output = torch.matmul(attention_prob,V)
        return output
    
    # Steps for multi headed attention
    #Split the heads
    def split_head(self, X):
        #Reshape the input to have num of heads for muti headed attention
        batch_size, seq_length, multi_model = X.size()
        return X.view(batch_size, seq_length,self.num_heads,self.d_k).transpose(1,2)
    
    #Combine the heads
    def combine_heads(self,X):
        #Combine multiple heads to original shape
        batch_size,_,seq_length,d_k = X.size()
        return X.transpose(1,2).contiguous().view(batch_size,seq_length, self.d_model)
    
    def forward(self, Q,V,K, mask=None):
        #Apply linear transformation and split heads
        Q = self.split_head(self.w_q(Q))
        K = self.split_head(self.w_k(K))
        V = self.split_head(self.w_v(V))

        #Perform scaled dot product attention
        attention_output = self.scaled_dot_product_attention(Q,K,V, mask)

        #Calculate the output and combine heads
        output = self.w_o(self.combine_heads(attention_output))
        return output
    
#Position Feed Forward Networks are applied to each token position independently
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model,d_ff):
        super(PositionWiseFeedForward,self).__init__()
        #d_model is dimentionality of the model and 
        # d_ff is the dimensionality of the inner layer in the feed forward network
        self.fc1 = nn.Linear(d_model,d_ff)
        self.fc2 = nn.Linear(d_ff,d_model)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        return self.fc2(self.relu(self.fc1(x)))
    
#Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding,self).__init__()

        #Create empty positional matrix
        pe = torch.zeros(max_seq_length,d_model)

        #Positional indices and it returns [max_seq_len,1]
        position = torch.arange(
            0,
            max_seq_length,
            dtype=torch.float
        ).unsqueeze(1)

        #Frequency scalling term, this creates different wavelengths/frequecies for dimensions
        freq = torch.exp(
            torch.arange(0,d_model,2).float()
            *
            -(math.log(10000.0)/d_model)
        )

        #Apply sine to the even positions
        pe[:,0::2] = torch.sin(freq*position)
        #Apply Cosine to the odd positions
        pe[:,1::2] = torch.cos(freq*position)

        #Register buffer stores positional encodings inside model
        self.register_buffer("pe", pe.unsqueeze(0))
    
    def forward(self,x):
        #It uses the first X element of pe to ensure that positional encodings match the actual sequence of length of X
        return x + self.pe[:, :x.size(1)]


    

