# MedCLIP_Inplanted.py (hierarchical multi-label)
import torch
import torch.nn as nn
from timm.models.swin_transformer import SwinTransformer

class ClipAdapter(nn.Module):
    def __init__(self, c_in, bottleneck):
        super(ClipAdapter, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(c_in, bottleneck, bias=False),
            nn.LeakyReLU(inplace=False)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(bottleneck, c_in, bias=False),
            nn.LeakyReLU(inplace=False)
        )

    def forward(self, x):
        x_intermediate = self.fc1(x)
        y_residual = self.fc2(x_intermediate)
        return x_intermediate, y_residual

class CLIP_Swin_Implanted(nn.Module):
    def __init__(self, image_size=224, text_embedding_dim=768):
        super().__init__()
        self.backbone = SwinTransformer(
            img_size=image_size,
            patch_size=4,
            in_chans=3,
            embed_dim=96,
            depths=(2, 2, 6, 2),
            num_heads=(3, 6, 12, 24),
            window_size=7,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_path_rate=0.2,
            norm_layer=nn.LayerNorm,
            num_classes=0
        )

        self.c_in_stages = [self.backbone.embed_dim * (2**i) for i in range(self.backbone.num_layers)]

        self.seg_adapters = nn.ModuleList([
            ClipAdapter(c, bottleneck=c//2) for c in self.c_in_stages
        ])
        self.cls_adapters = nn.ModuleList([
            ClipAdapter(c, bottleneck=c//2) for c in self.c_in_stages
        ])
        adapter_output_dim = self.c_in_stages[-1] // 2
        self.image_projection = nn.Sequential(
            nn.LayerNorm(adapter_output_dim),
            nn.Linear(adapter_output_dim, text_embedding_dim)
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)

    def forward(self, x_input_image):
        current_features = self.backbone.patch_embed(x_input_image)
        seg_intermediates = []
        cls_intermediates = []

        for i, stage in enumerate(self.backbone.layers):
            current_features = stage(current_features)
            seg_intermediate, seg_residual = self.seg_adapters[i](current_features)
            cls_intermediate, cls_residual = self.cls_adapters[i](current_features)

            B, H_s, W_s, bottle_neck_dim = seg_intermediate.shape
            seg_intermediate = seg_intermediate.view(B, H_s * W_s, bottle_neck_dim)
            cls_intermediate = cls_intermediate.view(B, H_s * W_s, bottle_neck_dim)
            
            seg_intermediates.append(seg_intermediate)
            cls_intermediates.append(cls_intermediate)

            current_features = current_features * 0.6 + seg_residual * 0.2 + cls_residual * 0.2

        last_cls_intermediate = cls_intermediates[-1]
        cls_feat = last_cls_intermediate.mean(dim=1)

        projected_image_feat = self.image_projection(cls_feat)
        projected_image_feat = projected_image_feat / projected_image_feat.norm(dim=-1, keepdim=True)
        return projected_image_feat, self.logit_scale, seg_intermediates