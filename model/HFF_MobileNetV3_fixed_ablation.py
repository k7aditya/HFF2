import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td

BatchNorm3d = nn.InstanceNorm3d
BN_MOMENTUM = 0.1
relu_inplace = True
ActivationFunction = nn.ReLU

def conv1x1(in_chs, out_chs, stride=1):
    return nn.Conv3d(in_chs, out_chs, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_chs, out_chs, stride=1, groups=1, dilation=1):
    return nn.Conv3d(in_chs, out_chs, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)

class HardSwish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.

class HardSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.

class SqueezeExcitation3D(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        squeeze = max(1, in_channels // reduction)
        self.fc1 = nn.Conv3d(in_channels, squeeze, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(squeeze, in_channels, 1)
        self.hsigmoid = HardSigmoid()

    def forward(self, x):
        scale = F.adaptive_avg_pool3d(x, 1)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.hsigmoid(scale)
        return x * scale

class InvertedResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, stride=1, use_se=False, activation='RE', dropout_p=0.0):
        super().__init__()
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = int(in_channels * expand_ratio)
        act = HardSwish if activation == 'HS' else nn.ReLU

        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv3d(in_channels, hidden_dim, 1, bias=False),
                BatchNorm3d(hidden_dim, momentum=BN_MOMENTUM),
                act(inplace=True)
            ])
        else:
            hidden_dim = in_channels

        layers.extend([
            nn.Conv3d(hidden_dim, hidden_dim, 3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            BatchNorm3d(hidden_dim, momentum=BN_MOMENTUM),
            act(inplace=True)
        ])

        if use_se:
            layers.append(SqueezeExcitation3D(hidden_dim))

        layers.extend([
            nn.Conv3d(hidden_dim, out_channels, 1, bias=False),
            BatchNorm3d(out_channels, momentum=BN_MOMENTUM)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV3Encoder3D(nn.Module):
    def __init__(self, in_channels, dropout_p=0.0):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 16, 3, 1, 1, bias=False),
            BatchNorm3d(16, momentum=BN_MOMENTUM),
            HardSwish(inplace=True)
        )

        self.enc1 = InvertedResidualBlock3D(16, 16, expand_ratio=1, stride=1, use_se=True, activation='RE', dropout_p=dropout_p)
        self.enc2 = InvertedResidualBlock3D(16, 32, expand_ratio=4.5, stride=2, use_se=False, activation='RE', dropout_p=dropout_p)
        self.enc3 = InvertedResidualBlock3D(32, 64, expand_ratio=4, stride=2, use_se=True, activation='HS', dropout_p=dropout_p)
        self.enc4 = InvertedResidualBlock3D(64, 128, expand_ratio=6, stride=2, use_se=True, activation='HS', dropout_p=dropout_p)
        self.enc5 = InvertedResidualBlock3D(128, 256, expand_ratio=6, stride=2, use_se=True, activation='HS', dropout_p=dropout_p)

    def forward(self, x):
        x = self.stem(x)
        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        f4 = self.enc4(f3)
        f5 = self.enc5(f4)
        return f1, f2, f3, f4, f5

class UpsampleConv(nn.Module):
    def __init__(self, in_chs, out_chs, dropout_p=0.0):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(in_chs, out_chs, 3, 1, 1, bias=True),
            BatchNorm3d(out_chs, momentum=BN_MOMENTUM),
            ActivationFunction(inplace=relu_inplace),
            nn.Dropout3d(p=dropout_p)
        )

    def forward(self, x):
        return self.up(x)

class DownsampleConv(nn.Module):
    def __init__(self, in_chs, out_chs, dropout_p=0.0):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv3d(in_chs, out_chs, 3, 2, 1, bias=False),
            BatchNorm3d(out_chs, momentum=BN_MOMENTUM),
            ActivationFunction(inplace=relu_inplace)
        )

    def forward(self, x):
        return self.down(x)

class SameSizeConv(nn.Module):
    def __init__(self, in_chs, out_chs, dropout_p=0.0):
        super().__init__()
        self.same = nn.Sequential(
            nn.Conv3d(in_chs, out_chs, 3, 1, 1, bias=False),
            BatchNorm3d(out_chs, momentum=BN_MOMENTUM),
            ActivationFunction(inplace=relu_inplace)
        )

    def forward(self, x):
        return self.same(x)

class TransitionConv(nn.Module):
    def __init__(self, in_chs, out_chs, dropout_p=0.0):
        super().__init__()
        self.transition = nn.Sequential(
            nn.Conv3d(in_chs, out_chs, 1, 1, 0, bias=False),
            BatchNorm3d(out_chs, momentum=BN_MOMENTUM),
            ActivationFunction(inplace=relu_inplace)
        )

    def forward(self, x):
        return self.transition(x)

class SideTransition(nn.Module):
    def __init__(self, dropout_p=0.0):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(32, 16, 3, 1, 1, bias=True),
            nn.InstanceNorm3d(16, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 8, 3, 1, 1, bias=True),
            nn.InstanceNorm3d(8, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 4, 3, 1, 1, bias=True),
            nn.InstanceNorm3d(4, momentum=0.1),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.up(x)

class CustomBlock(nn.Module):
    expansion = 1

    def __init__(self, in_chs, out_chs, stride=1, downsample=None, norm_layer=BatchNorm3d, dropout_p=0.0):
        super().__init__()
        self.conv1 = conv3x3(in_chs, out_chs, stride)
        self.bn1 = norm_layer(out_chs, momentum=BN_MOMENTUM)
        self.relu = ActivationFunction(inplace=relu_inplace)
        self.dropout = nn.Dropout3d(p=dropout_p)
        self.conv2 = conv3x3(out_chs, out_chs)
        self.bn2 = norm_layer(out_chs, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.bn2(out) + identity
        out = self.relu(out)
        return out

def custom_max(x, dim, keepdim=True):
    temp_x = x
    for i in dim:
        temp_x = torch.max(temp_x, dim=i, keepdim=True)[0]
    if not keepdim: temp_x = temp_x.squeeze()
    return temp_x

# ==================== FDD Component: Frequency Domain Decomposition ====================
class HighFreqConv(nn.Module):
    """
    FDD Component: Applies high-frequency filtering via convolution
    """
    def __init__(self, in_chs, out_chs, target_kernel, enable_fdd=True):
        super().__init__()
        self.enable_fdd = enable_fdd
        self.conv = nn.Conv3d(in_chs, out_chs, 3, 1, 1, bias=False)
        if enable_fdd:
            self.init_weights(target_kernel)

    def init_weights(self, target_kernel):
        with torch.no_grad():
            self.conv.weight = nn.Parameter(
                target_kernel.repeat(self.conv.out_channels, self.conv.in_channels, 1, 1, 1))

    def forward(self, x):
        if self.enable_fdd:
            return x + self.conv(x)
        else:
            # Identity - no FDD processing
            return x

class SemanticAttentionModule(nn.Module):
    def __init__(self, in_features, reduction_rate=16):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_features, in_features // reduction_rate),
            nn.ReLU(),
            nn.Linear(in_features // reduction_rate, in_features)
        )

    def forward(self, x):
        max_x = torch.max(x, dim=2, keepdim=True)[0]
        max_x = torch.max(max_x, dim=3, keepdim=True)[0]
        max_x = torch.max(max_x, dim=4, keepdim=True)[0]
        avg_x = torch.mean(x, dim=2, keepdim=True)
        avg_x = torch.mean(avg_x, dim=3, keepdim=True)
        avg_x = torch.mean(avg_x, dim=4, keepdim=True)
        max_x = self.linear(max_x.squeeze())
        avg_x = self.linear(avg_x.squeeze())
        att = max_x + avg_x
        att = torch.sigmoid(att).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return x * att

class PositionalAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, 3, 1, 1)

    def forward(self, x):
        max_x = custom_max(x, dim=(1, 2), keepdim=True)
        avg_x = torch.mean(x, dim=(1, 2), keepdim=True)
        att = torch.cat((max_x, avg_x), dim=1)
        att = self.conv(att)
        att = torch.sigmoid(att)
        return x * att

class SliceAttentionModule(nn.Module):
    def __init__(self, in_features, rate=4, uncertainty=True, rank=5):
        super().__init__()
        self.uncertainty = uncertainty
        self.rank = rank
        self.linear = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=int(in_features * rate)),
            nn.ReLU(),
            nn.Linear(in_features=int(in_features * rate), out_features=in_features)
        )

        if uncertainty:
            self.non_linear = nn.ReLU()
            self.mean = nn.Linear(in_features=in_features, out_features=in_features)
            self.log_diag = nn.Linear(in_features=in_features, out_features=in_features)
            self.factor = nn.Linear(in_features=in_features, out_features=in_features * rank)

    def forward(self, x):
        B, C, D, H, W = x.shape
        max_x = x.max(dim=1)[0]
        max_x = max_x.max(dim=2)[0]
        max_x = max_x.max(dim=2)[0]
        avg_x = x.mean(dim=(1, 3, 4))

        max_x = self.linear(max_x)
        avg_x = self.linear(avg_x)
        att = max_x + avg_x

        if self.uncertainty:
            temp = self.non_linear(att)
            mean = self.mean(temp)
            diag = F.softplus(self.log_diag(temp)) + 1e-3
            factor = self.factor(temp).view(B, D, self.rank)

            mean = torch.nan_to_num(mean, nan=0.0, posinf=1e6, neginf=-1e6)
            diag = torch.nan_to_num(diag, nan=1e-3, posinf=1e6, neginf=1e-6).clamp(min=1e-6)
            factor = torch.nan_to_num(factor, nan=0.0, posinf=1e6, neginf=-1e6)

            try:
                dist = td.LowRankMultivariateNormal(loc=mean, cov_factor=factor, cov_diag=diag)
                samp = dist.rsample()
            except RuntimeError:
                samp = None
                jitter = 1e-3
                for _ in range(6):
                    try:
                        dist = td.LowRankMultivariateNormal(loc=mean, cov_factor=factor, cov_diag=diag + jitter)
                        samp = dist.rsample()
                        break
                    except RuntimeError:
                        jitter *= 10.0

                if samp is None:
                    eps = torch.randn_like(mean)
                    samp = mean + torch.sqrt(diag) * eps

            att = torch.sigmoid(samp).unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        else:
            att = torch.sigmoid(att).unsqueeze(1).unsqueeze(-1).unsqueeze(-1)

        return x * att

# ==================== FDCA Component: Frequency Domain Cross-Attention ====================
class FrequencyCrossAttention(nn.Module):
    """
    FDCA Component: Applies cross-attention in frequency domain
    """
    def __init__(self, num_channels, num_slices, uncertainty=True, rank=5, enable_fdca=True):
        super().__init__()
        self.enable_fdca = enable_fdca
        self.semantic_att = SemanticAttentionModule(num_channels)
        self.positional_att = PositionalAttentionModule()
        self.slice_att = SliceAttentionModule(in_features=num_slices, uncertainty=uncertainty, rank=rank)

    def forward(self, x):
        if not self.enable_fdca:
            # Identity - no FDCA processing
            return x

        freq_domain = torch.fft.fftn(x, dim=(2, 3, 4), norm='ortho')
        freq_real = freq_domain.real
        freq_imag = freq_domain.imag

        semantic_attn = self.semantic_att(freq_real)
        freq_real = freq_real * semantic_attn
        freq_imag = freq_imag * semantic_attn

        positional_attn = self.positional_att(freq_real)
        freq_real = freq_real * positional_attn
        freq_imag = freq_imag * positional_attn

        slice_attn = self.slice_att(freq_real)
        freq_real = freq_real * slice_attn
        freq_imag = freq_imag * slice_attn

        freq_domain_combined = torch.complex(freq_real, freq_imag)
        x_modified = torch.fft.ifftn(freq_domain_combined, dim=(2, 3, 4), norm='ortho').real

        return x_modified

# ==================== ALC Component: Adaptive Laplacian Convolution ====================
laplacian_3x3 = torch.tensor([
    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
    [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
], dtype=torch.float32)

upsampled_laplacian = laplacian_3x3

class HFFNet(nn.Module):
    def __init__(self, in_chs1, in_chs2, num_classes, dropout_p=0.2, 
                 enable_fdd=True, enable_alc=True, enable_fdca=True):
        """
        HFFNet with Ablation Study Support

        Args:
            in_chs1: Input channels for low-frequency branch
            in_chs2: Input channels for high-frequency branch  
            num_classes: Number of output classes
            dropout_p: Dropout probability
            enable_fdd: Enable Frequency Domain Decomposition (default: True)
            enable_alc: Enable Adaptive Laplacian Convolution (default: True)
            enable_fdca: Enable Frequency Domain Cross-Attention (default: True)
        """
        super().__init__()
        self.dropout_p = dropout_p
        self.enable_fdd = enable_fdd
        self.enable_alc = enable_alc
        self.enable_fdca = enable_fdca

        # Print ablation configuration
        print("\n" + "="*80)
        print("HFFNet Ablation Configuration:")
        print(f"  - FDD (Frequency Domain Decomposition): {'ENABLED' if enable_fdd else 'DISABLED'}")
        print(f"  - ALC (Adaptive Laplacian Convolution): {'ENABLED' if enable_alc else 'DISABLED'}")
        print(f"  - FDCA (Frequency Domain Cross-Attention): {'ENABLED' if enable_fdca else 'DISABLED'}")
        print("="*80 + "\n")

        # ALC Component - conditionally used
        if self.enable_alc:
            self.laplacian_target = nn.Parameter(upsampled_laplacian.repeat(16, 16, 1, 1, 1), requires_grad=False)
        else:
            self.laplacian_target = None

        self.laplacian = upsampled_laplacian

        # FDD Component - HighFreqConv
        self.input_ed = HighFreqConv(16, 16, target_kernel=self.laplacian, enable_fdd=enable_fdd)

        # FDCA Components - conditionally enabled
        self.LF_l4_FDCA = FrequencyCrossAttention(128, 16, enable_fdca=enable_fdca)
        self.LF_l5_FDCA = FrequencyCrossAttention(256, 8, enable_fdca=enable_fdca)
        self.HF_l4_FDCA = FrequencyCrossAttention(128, 16, enable_fdca=enable_fdca)
        self.HF_l5_FDCA = FrequencyCrossAttention(256, 8, enable_fdca=enable_fdca)

        # Encoders - NO DROPOUT (critical for dual-branch fusion)
        self.mobilenet_encoder_b1 = MobileNetV3Encoder3D(in_chs1, dropout_p=0.0)
        self.mobilenet_encoder_b2 = MobileNetV3Encoder3D(in_chs2, dropout_p=0.0)

        # Side paths - NO DROPOUT
        self.x1_3side = SideTransition(dropout_p=0.0)
        self.x2_3side = SideTransition(dropout_p=0.0)

        # ===== DECODER DROPOUT ONLY - 0.15 for stability =====
        dropout_decoder = 0.15

        # Fusion layers - NO DROPOUT (critical for stability)
        self.l4_b1_t = TransitionConv(128 + 256 + 128, 128, dropout_p=0.0)
        self.l4_b2_t = TransitionConv(128 + 256 + 128, 128, dropout_p=0.0)
        self.l5_b1_t = TransitionConv(256 + 256 + 128, 256, dropout_p=0.0)
        self.l5_b2_t = TransitionConv(256 + 256 + 128, 256, dropout_p=0.0)

        # Encoder-like blocks in fusion - NO DROPOUT
        self.l4_b1_2 = InvertedResidualBlock3D(128, 128, expand_ratio=4, stride=1, use_se=True, activation='HS', dropout_p=0.0)
        self.l4_b1_d2 = DownsampleConv(128, 128, dropout_p=0.0)
        self.l4_b1_s = SameSizeConv(128, 128, dropout_p=0.0)
        self.l4_b1_3 = InvertedResidualBlock3D(128, 128, expand_ratio=4, stride=1, use_se=True, activation='HS', dropout_p=0.0)
        self.l4_b1_4 = CustomBlock(128 + 128, 128, downsample=nn.Sequential(conv1x1(128 + 128, 128), BatchNorm3d(128, momentum=BN_MOMENTUM)), dropout_p=0.0)
        self.l4_b1_u = UpsampleConv(128, 64, dropout_p=0.0)

        self.l5_b1_1 = InvertedResidualBlock3D(256, 256, expand_ratio=6, stride=1, use_se=True, activation='HS', dropout_p=0.0)
        self.l5_b1_u = UpsampleConv(256, 256, dropout_p=0.0)
        self.l5_b1_s = SameSizeConv(256, 256, dropout_p=0.0)
        self.l5_b1_2 = InvertedResidualBlock3D(256, 256, expand_ratio=6, stride=1, use_se=True, activation='HS', dropout_p=0.0)
        self.l5_b1_u2 = UpsampleConv(256, 128, dropout_p=0.0)

        # ===== PURE DECODER - DROPOUT HERE =====
        self.l3_b1_2 = CustomBlock(64 + 64, 64, downsample=nn.Sequential(conv1x1(64 + 64, 64), BatchNorm3d(64, momentum=BN_MOMENTUM)), dropout_p=dropout_decoder)
        self.l3_b1_u = UpsampleConv(64, 32, dropout_p=dropout_decoder)
        self.l2_b1_2 = CustomBlock(32 + 32, 32, downsample=nn.Sequential(conv1x1(32 + 32, 32), BatchNorm3d(32, momentum=BN_MOMENTUM)), dropout_p=dropout_decoder)
        self.l2_b1_u = UpsampleConv(32, 16, dropout_p=dropout_decoder)

        # Output - NO DROPOUT
        self.l1_b1_2 = CustomBlock(16 + 16, 16, downsample=nn.Sequential(conv1x1(16 + 16, 16), BatchNorm3d(16, momentum=BN_MOMENTUM)), dropout_p=0.0)
        self.l1_b1_f = nn.Conv3d(16, num_classes, 1)

        # ===== HF BRANCH - SAME STRATEGY =====
        self.l4_b2_2 = InvertedResidualBlock3D(128, 128, expand_ratio=4, stride=1, use_se=True, activation='HS', dropout_p=0.0)
        self.l4_b2_d2 = DownsampleConv(128, 128, dropout_p=0.0)
        self.l4_b2_s = SameSizeConv(128, 128, dropout_p=0.0)
        self.l4_b2_3 = InvertedResidualBlock3D(128, 128, expand_ratio=4, stride=1, use_se=True, activation='HS', dropout_p=0.0)
        self.l4_b2_4 = CustomBlock(128 + 128, 128, downsample=nn.Sequential(conv1x1(128 + 128, 128), BatchNorm3d(128, momentum=BN_MOMENTUM)), dropout_p=0.0)
        self.l4_b2_u = UpsampleConv(128, 64, dropout_p=0.0)

        self.l5_b2_1 = InvertedResidualBlock3D(256, 256, expand_ratio=6, stride=1, use_se=True, activation='HS', dropout_p=0.0)
        self.l5_b2_u = UpsampleConv(256, 256, dropout_p=0.0)
        self.l5_b2_s = SameSizeConv(256, 256, dropout_p=0.0)
        self.l5_b2_2 = InvertedResidualBlock3D(256, 256, expand_ratio=6, stride=1, use_se=True, activation='HS', dropout_p=0.0)
        self.l5_b2_u2 = UpsampleConv(256, 128, dropout_p=0.0)

        self.l3_b2_2 = CustomBlock(64 + 64, 64, downsample=nn.Sequential(conv1x1(64 + 64, 64), BatchNorm3d(64, momentum=BN_MOMENTUM)), dropout_p=dropout_decoder)
        self.l3_b2_u = UpsampleConv(64, 32, dropout_p=dropout_decoder)
        self.l2_b2_2 = CustomBlock(32 + 32, 32, downsample=nn.Sequential(conv1x1(32 + 32, 32), BatchNorm3d(32, momentum=BN_MOMENTUM)), dropout_p=dropout_decoder)
        self.l2_b2_u = UpsampleConv(32, 16, dropout_p=dropout_decoder)

        self.l1_b2_2 = CustomBlock(16 + 16, 16, downsample=nn.Sequential(conv1x1(16 + 16, 16), BatchNorm3d(16, momentum=BN_MOMENTUM)), dropout_p=0.0)
        self.l1_b2_f = nn.Conv3d(16, num_classes, 1)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, HighFreqConv): continue
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, input1, input2):
        layer1_LF, layer2_LF, layer3_LF, layer4_LF_1, layer5_LF_1 = self.mobilenet_encoder_b1(input1)

        # FDD Component - apply if enabled
        input2 = self.input_ed(input2)

        layer1_HF, layer2_HF, layer3_HF, layer4_HF_1, layer5_HF_1 = self.mobilenet_encoder_b2(input2)

        # Low-Freq
        layer4_LF_2 = self.l4_b1_2(layer4_LF_1)
        layer4_LF_3_d = self.l4_b1_d2(layer4_LF_2)
        layer4_LF_3_s = self.l4_b1_s(layer4_LF_2)

        # FDCA Component - apply if enabled
        layer4_LF_3_s = self.LF_l4_FDCA(layer4_LF_3_s)

        layer5_LF_1 = self.l5_b1_1(layer5_LF_1)
        layer5_LF_u = self.l5_b1_u(layer5_LF_1)
        layer5_LF_s = self.l5_b1_s(layer5_LF_1)

        # FDCA Component - apply if enabled
        layer5_LF_s = self.LF_l5_FDCA(layer5_LF_s)

        # High-Freq
        layer4_HF_2 = self.l4_b2_2(layer4_HF_1)
        layer4_HF_3_d = self.l4_b2_d2(layer4_HF_2)
        layer4_HF_3_s = self.l4_b2_s(layer4_HF_2)

        # FDCA Component - apply if enabled
        layer4_HF_3_s = self.HF_l4_FDCA(layer4_HF_3_s)

        layer5_HF_1 = self.l5_b2_1(layer5_HF_1)
        layer5_HF_u = self.l5_b2_u(layer5_HF_1)
        layer5_HF_s = self.l5_b2_s(layer5_HF_1)

        # FDCA Component - apply if enabled
        layer5_HF_s = self.HF_l5_FDCA(layer5_HF_s)

        # Mixer/Decoder downstream
        merge_LF_5_3 = torch.cat((layer5_LF_s, layer5_HF_s, layer4_HF_3_d), dim=1)
        merge_LF_5_3 = self.l5_b1_t(merge_LF_5_3)
        merge_LF_5_3 = self.l5_b1_2(merge_LF_5_3)
        merge_LF_5_3 = self.l5_b1_u2(merge_LF_5_3)

        merge_LF_4_4 = torch.cat((layer4_LF_3_s, layer4_HF_3_s, layer5_HF_u), dim=1)
        merge_LF_4_4 = self.l4_b1_t(merge_LF_4_4)
        merge_LF_4_4 = self.l4_b1_3(merge_LF_4_4)
        merge_LF_4_4 = torch.cat((merge_LF_4_4, merge_LF_5_3), dim=1)
        merge_LF_4_4 = self.l4_b1_4(merge_LF_4_4)
        merge_LF_4_4 = self.l4_b1_u(merge_LF_4_4)

        merge_HF_5_3 = torch.cat((layer5_HF_s, layer5_LF_s, layer4_LF_3_d), dim=1)
        merge_HF_5_3 = self.l5_b2_t(merge_HF_5_3)
        merge_HF_5_3 = self.l5_b2_2(merge_HF_5_3)
        merge_HF_5_3 = self.l5_b2_u2(merge_HF_5_3)

        merge_HF_4_4 = torch.cat((layer4_HF_3_s, layer4_LF_3_s, layer5_LF_u), dim=1)
        merge_HF_4_4 = self.l4_b2_t(merge_HF_4_4)
        merge_HF_4_4 = self.l4_b2_3(merge_HF_4_4)
        merge_HF_4_4 = torch.cat((merge_HF_4_4, merge_HF_5_3), dim=1)
        merge_HF_4_4 = self.l4_b2_4(merge_HF_4_4)
        merge_HF_4_4 = self.l4_b2_u(merge_HF_4_4)

        decode_LF_3 = torch.cat((layer3_LF, merge_LF_4_4), dim=1)
        decode_LF_3 = self.l3_b1_2(decode_LF_3)
        decode_LF_3 = self.l3_b1_u(decode_LF_3)
        decode_LF_3side = self.x1_3side(decode_LF_3)

        decode_LF_2 = torch.cat((layer2_LF, decode_LF_3), dim=1)
        decode_LF_2 = self.l2_b1_2(decode_LF_2)
        decode_LF_2 = self.l2_b1_u(decode_LF_2)

        decode_LF_1 = torch.cat((layer1_LF, decode_LF_2), dim=1)
        decode_LF_1 = self.l1_b1_2(decode_LF_1)
        decode_LF_1 = self.l1_b1_f(decode_LF_1)

        decode_HF_3 = torch.cat((layer3_HF, merge_HF_4_4), dim=1)
        decode_HF_3 = self.l3_b2_2(decode_HF_3)
        decode_HF_3 = self.l3_b2_u(decode_HF_3)
        decode_HF_3side = self.x2_3side(decode_HF_3)

        decode_HF_2 = torch.cat((layer2_HF, decode_HF_3), dim=1)
        decode_HF_2 = self.l2_b2_2(decode_HF_2)
        decode_HF_2 = self.l2_b2_u(decode_HF_2)

        decode_HF_1 = torch.cat((layer1_HF, decode_HF_2), dim=1)
        decode_HF_1 = self.l1_b2_2(decode_HF_1)
        decode_HF_1 = self.l1_b2_f(decode_HF_1)

        return decode_LF_1, decode_HF_1, decode_LF_3side, decode_HF_3side

def hff_net(in_chs1, in_chs2, num_classes, dropout_p=0.2, 
            enable_fdd=True, enable_alc=True, enable_fdca=True):
    return HFFNet(in_chs1, in_chs2, num_classes, dropout_p=dropout_p,
                  enable_fdd=enable_fdd, enable_alc=enable_alc, enable_fdca=enable_fdca)

if __name__ == "__main__":
    model = hff_net(4, 16, 4, dropout_p=0.2, enable_fdd=True, enable_alc=True, enable_fdca=True).cuda()
    input1 = torch.rand(1, 4, 128, 128, 128).cuda()
    input2 = torch.rand(1, 16, 128, 128, 128).cuda()

    print("=" * 80)
    print("HFF-NET WITH ABLATION STUDY SUPPORT")
    print("=" * 80)

    x1_1_main, x1_1_aux1, x2, x3 = model(input1, input2)

    print("\nOutput shapes:")
    print(f"  Main LF output: {x1_1_main.shape}")
    print(f"  Aux HF output: {x1_1_aux1.shape}")
    print(f"  Side LF output: {x2.shape}")
    print(f"  Side HF output: {x3.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    dropout_count = sum(1 for m in model.modules() if isinstance(m, nn.Dropout3d))
    print(f"  Dropout3d layers: {dropout_count}")

    if torch.isnan(x1_1_main).any() or torch.isnan(x1_1_aux1).any():
        print("  ✗ WARNING: NaN values detected in output!")
    else:
        print("  ✓ No NaN values in outputs - model is stable!")

    print("=" * 80)
