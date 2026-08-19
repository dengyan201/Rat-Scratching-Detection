import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights

class TemporalAttention(nn.Module):
    """
    Self-attention along time independently at each spatial location.
    Applied only at later R3D stages to control GPU memory.
    """
    def __init__(self,channels,heads,dropout=0.1):
        super().__init__()
        self.n1=nn.LayerNorm(channels)
        self.attn=nn.MultiheadAttention(channels,heads,dropout=dropout,batch_first=True)
        self.n2=nn.LayerNorm(channels)
        self.ff=nn.Sequential(
            nn.Linear(channels,channels*2),nn.GELU(),nn.Dropout(dropout),
            nn.Linear(channels*2,channels),nn.Dropout(dropout)
        )

    def forward(self,x):
        b,c,t,h,w=x.shape
        z=x.permute(0,3,4,2,1).contiguous().view(b*h*w,t,c)
        q=self.n1(z)
        a,_=self.attn(q,q,q,need_weights=False)
        z=z+a
        z=z+self.ff(self.n2(z))
        return z.view(b,h,w,t,c).permute(0,4,3,1,2).contiguous()

class ScratchR3DAttention(nn.Module):
    def __init__(self,pretrained=True,dropout=0.3):
        super().__init__()
        weights=R3D_18_Weights.DEFAULT if pretrained else None
        r=r3d_18(weights=weights)
        self.stem=r.stem
        self.layer1=r.layer1
        self.layer2=r.layer2
        self.layer3=r.layer3
        self.attn1=TemporalAttention(256,8)
        self.layer4=r.layer4
        self.attn2=TemporalAttention(512,8)
        self.pool=nn.AdaptiveAvgPool3d(1)
        self.fc=nn.Sequential(nn.Flatten(),nn.Dropout(dropout),nn.Linear(512,2))

    def forward(self,x):
        x=self.stem(x)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.attn1(x)
        x=self.layer4(x)
        x=self.attn2(x)
        return self.fc(self.pool(x))
