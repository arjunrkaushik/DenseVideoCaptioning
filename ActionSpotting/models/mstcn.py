import torch
import torch.nn as nn


class MSTCN2(nn.Module):
    def __init__(self, in_dim, num_f_maps, out_dim, num_layers, dropout=0.5, dilation_factor=2, ngroup=1, ln=False,
        in_map=True,
    ):
        super().__init__()
        assert ln == False

        self.num_layers = num_layers

        self.in_map = in_map
        if self.in_map:
            self.conv_1x1_in = nn.Conv1d(in_dim, num_f_maps, 1)
        else:
            assert in_dim == num_f_maps

        self.conv_dilated_1 = nn.ModuleList((
            nn.Conv1d(num_f_maps, num_f_maps, 3, padding=dilation_factor**(num_layers-1-i), dilation=dilation_factor**(num_layers-1-i), groups=ngroup)
            for i in range(num_layers)
        ))

        self.conv_dilated_2 = nn.ModuleList((
            nn.Conv1d(num_f_maps, num_f_maps, 3, padding=dilation_factor**i, dilation=dilation_factor**i, groups=ngroup)
            for i in range(num_layers)
        ))

        self.conv_fusion = nn.ModuleList((
             nn.Conv1d(2*num_f_maps, num_f_maps, 1)
             for i in range(num_layers)
            ))

        self.dropout = nn.Dropout(dropout)
        self.conv_out = nn.Conv1d(num_f_maps, out_dim, 1)

        self.string = f"MSTCN2(h:{in_dim}->{num_f_maps}x{num_layers}->{out_dim}, d={dilation_factor}, ng={ngroup}, dropout={dropout}, in_map={in_map})"

    def __str__(self):
        return self.string 

    def __repr__(self):
        return str(self)

    def forward(self, x):
        x = x.permute([1, 2, 0]) # 1, H, T

        if self.in_map:
            f = self.conv_1x1_in(x)
        else:
            f = x

        for i in range(self.num_layers):
            f_in = f
            f = self.conv_fusion[i](torch.cat([self.conv_dilated_1[i](f), self.conv_dilated_2[i](f)], 1))
            f = F.relu(f)
            if i != self.num_layers - 1:
                f = self.dropout(f)
            f = f + f_in

        out = self.conv_out(f)
        out = out.permute([2, 0, 1]) # T, 1, H 
        return out