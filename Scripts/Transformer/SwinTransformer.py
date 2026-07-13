import torch
import torch.nn as nn
from config import AudioConfig

#Define window partition
def window_partition(x,win):
    B,H,W,C = x.shape
    x = x.view(B,H//win, win, W//win, win,C)
    x = x.permute(0,1,3,2,4,5)
    x = x.reshape(-1,win,win,C)
    return x
    
#Define window reverse to generate tokens for the images
def window_reverse(windows, win, H,W):
    B = windows.shape[0]//(H//win * W//win)

    #Set the channel to -1 as it changes based on different transformer
    x = windows.view(B,H//win, W//win, win,win,-1)
    x = x.permute(0,1,3,2,4,5).contiguous()
    x = x.reshape(B,H,W,-1)
    return x

#Calculate window attention
class WindowAttention(nn.Module):
    def __init__(self,
                 d_model,
                 n_heads,
                 win,
                 dropout_rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = self.d_model // self.n_heads
        self.scale = self.head_dim**-0.5
        self.win = win

        #Define Q,K,V
        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout_rate)

        #Projection layer
        self.proj = nn.Linear(d_model,d_model)

        self.proj_drop = nn.Dropout(dropout_rate)

        #Relative Position bias this creates a table for Q and K indices
        coords = torch.stack(torch.meshgrid(torch.arange(win), torch.arange(win), 
                                            indexing="ij"))
        coords_flatten = coords.flatten(1)
        
        #Calculate the relative position
        rel = coords_flatten[:,:,None] - coords_flatten[:,None,:]
        rel = rel.permute(1,2,0)

        #Delta X
        rel[:,:,0] = rel[:,:,0] + (win-1)

        #Delta Y
        rel[:,:,1] = rel[:,:,1] + (win-1)

        #Calculate Indexes
        rel[:,:,0] = rel[:,:,0] * (2*win-1)
        index = rel.sum(-1)

        self.register_buffer("pos_index", index)
        self.rel_bias = nn.Parameter(torch.zeros((2*win-1) * (2*win - 1), n_heads))
    
    def forward(self,x, mask=None):
        B_,N,C = x.shape
        q = self.Q(x).reshape(B_, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.K(x).reshape(B_, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.V(x).reshape(B_, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # q = q * self.scale
        attn = (q @ k.transpose(-2,-1)) * self.scale
        rb = self.rel_bias[self.pos_index.view(-1)] .view(N,N,-1)
        attn = attn + rb.permute(2,0,1).unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(B_//nw, nw, self.n_heads, N,N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.n_heads, N,N)
        
        attn = torch.softmax(attn,dim=-1)
        attn = self.attn_dropout(attn)
        out = (attn @ v).transpose(1,2).reshape(B_,N,C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out
    
class SwinBlock(nn.Module):
    def __init__(self,
                 d_model,
                 res,
                 win,
                 shift,
                 n_heads,
                 diff,
                 dropout_rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.res = (int(res[0]), int(res[1]))
        self.win = win
        self.shift = shift

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = WindowAttention(d_model,n_heads,win,dropout_rate=dropout_rate/2)

        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model,diff),
            nn.GELU(),
            nn.Linear(diff,d_model),
            nn.Dropout(dropout_rate)
        )

        H,W = res
        if shift > 0:
            self.register_buffer("mask", self.create_mask(H,W,win,shift))
        else:
            self.mask = None
    
    def create_mask(self,H,W, win, shift):
        img_mask = torch.zeros((1,H,W,1))
        count = 0

        for h in (slice(0,-win), slice(-win,-shift), slice(-shift, None)):
            for w in (slice(0,-win), slice(-win,-shift), slice(-shift, None)):
                img_mask[:,h,w,:] = count
                count +=1
        
        mask = window_partition(img_mask, win)
        mask = mask.view(-1,win*win)
        mask = mask.unsqueeze(1) - mask.unsqueeze(2)
        mask = mask.masked_fill(mask!=0, -100000.0)
        return mask
    
    def forward(self,x):
        #L is total number of tokens
        B,L,C = x.shape

        H,W = int(self.res[0]), int(self.res[1])

        residual = x
        x = self.norm1(x)
        x = x.view(B,H,W,C)

        #Cyclic Shifting
        if self.shift > 0:
            x = torch.roll(x, shifts=(-self.shift, -self.shift), dims=(1,2))
        
        win_x = window_partition(x, self.win).view(-1,self.win*self.win, C)
        attn_out = self.attn(win_x, self.mask if self.mask is not None else None)
        x = window_reverse(attn_out,self.win,H,W)

        #Reverse shift to not have shift anymore
        if self.shift > 0:
            x = torch.roll(x, shifts=(+self.shift, +self.shift), dims=(1,2))
        
        x = residual + x.view(B,L,C)

        residual2 = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual2 + x
        return x
    
#Patch Merging
class PatchMerging(nn.Module):
    def __init__(self,
                 d_model):
        super().__init__()
        self.d_model = d_model

        self.reduction = nn.Linear(4*d_model, 2*d_model, bias=False)
        self.norm = nn.LayerNorm(4*d_model)

    def forward(self, x, H,W):
        B,L,C = x.shape
        x = x.view(B,H,W,C)

        #Extract four patches
        x0 = x[:,0::2,0::2,:].reshape(B,-1,C)
        x1 = x[:,1::2,0::2,:].reshape(B,-1,C)
        x2 = x[:,0::2,1::2,:].reshape(B,-1,C)
        x3 = x[:,1::2,1::2,:].reshape(B,-1,C)

        #Concat these batches
        x = torch.cat([x0,x1,x2,x3], dim=-1)
        # x = x.view(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)
        return x

#2 stage Swin Transformer
class Swin(nn.Module):
    def __init__(self,
                 config:AudioConfig, 
                 dropout_rate=0.2):
        super().__init__()
        self.architecture_name = "SwinTransformer"
        self.d_model = config.d_model
        win = 8
        self.patch_embed = nn.Conv2d(1,self.d_model, kernel_size=4,stride=4)
        initial_res = (32,32)

        #Stage 1
        self.stage1_block = nn.Sequential(
            #No shift swin block
            SwinBlock(d_model=self.d_model,
                      shift=0,
                      n_heads=config.n_heads,
                      diff=config.d_ff,
                      win=win,
                      res=initial_res),

            #Shift swin block
            SwinBlock(d_model=self.d_model,
                      shift=win//2,
                      n_heads=config.n_heads,
                      diff=config.d_ff,
                      win=win,
                      res=initial_res)
        )

        #Patch merging
        self.patch_merge = PatchMerging(self.d_model)
        stage2_res = (initial_res[0]//2,initial_res[1]//2)
        merged_dim = self.d_model*2

        #Stage 2
        self.stage2_block = nn.Sequential(
            #No shift swin block
            SwinBlock(d_model=merged_dim,
                      shift=0,
                      n_heads=config.n_heads,
                      diff=config.d_ff,
                      win=win//2,
                      res=stage2_res),

            #Shift swin block
            SwinBlock(d_model=merged_dim,
                      shift=win//4,
                      n_heads=config.n_heads,
                      diff=config.d_ff*2,
                      win=win//2,
                      res=stage2_res)
            )
       
        self.norm = nn.LayerNorm(merged_dim)
        self.fc_drop = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(merged_dim, config.num_classes)
    
    def forward(self,x):
        x = self.patch_embed(x)
        B,C,H,W = x.shape
        x = x.flatten(2).transpose(1,2)
        #Number of tokens
        L = H*W

        #Stage 1 transformer
        x = self.stage1_block(x)

        #Patch merging
        x = self.patch_merge(x,H,W)
        H,W = H//2, W//2
        

        #Stage 2 transformer
        x = self.stage2_block(x)

        #Classification layer
        x = self.norm(x)
        x = x.mean(1) #Global average Pooling

        x = self.fc_drop(x) #Apply structural dropout before final projection classification
        x = self.fc(x)

        return x




















