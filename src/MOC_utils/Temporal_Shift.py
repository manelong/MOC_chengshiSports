import torch
import torch.nn as nn

class TemporalShift(object):
    def __init__(self, n_segment=3, n_div=8):
        self.n_segment = n_segment
        self.fold_div = n_div

    def __call__(self, x):
        return self.shift(x, self.n_segment, self.fold_div)
    
    def shift(self, x, n_segment, fold_div=3):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div

        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

        return out.view(nt, c, h, w)

class LightweightTemporalAttentionModule(nn.Module):
    def __init__(self, num_channels, reduction_ratio=4):
        super(LightweightTemporalAttentionModule, self).__init__()
        self.num_channels = num_channels
        self.reduced_channels = num_channels // reduction_ratio
        
        # 全连接层（downscale 和 upscale）
        self.fc1 = nn.Linear(in_features=num_channels, out_features=self.reduced_channels, bias=False)
        self.fc2 = nn.Linear(in_features=self.reduced_channels, out_features=num_channels, bias=False)
        
        # 激活函数
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x 的形状：(B, T, C, H, W)
        B, T, C, H, W = x.size()
        
        # 计算空间上 (H, W) 的均值来进行时间池化
        avg_pool = x.view(B, T, C, -1).mean(dim=-1)  # 形状：(B, T, C)
        
        # 通过全连接层
        fc1 = self.relu(self.fc1(avg_pool))  # 形状：(B, T, C // reduction_ratio)
        fc2 = self.fc2(fc1)  # 形状：(B, T, C)
        
        # 通过 Sigmoid 函数
        attention = self.sigmoid(fc2).view(B, T, C, 1, 1)  # 形状：(B, T, C, 1, 1)
        
        # 应用注意力
        out = x * attention  # 注意力机制
        
        return out
    
if __name__ == "__main__":
    B, T, C, H, W = 2, 8, 64, 32, 32  # 示例维度
    x = torch.randn(B, T, C, H, W)  # 示例输入
    attention_module = LightweightTemporalAttentionModule(num_channels=C)
    
    output = attention_module(x)
    print(output.shape)  # 输出应与输入形状相同: (B, T, C, H, W)