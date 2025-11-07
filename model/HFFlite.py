import torch
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as F
from einops import rearrange
 
BatchNorm3d = nn.InstanceNorm3d
BN_MOMENTUM = 0.1

relu_inplace = True
ActivationFunction = nn.ReLU

def conv1x1(in_chs, out_chs, stride=1):
    """1x1 convolution"""
    return nn.Conv3d(in_chs, out_chs, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_chs, out_chs, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_chs, out_chs, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)

class UpsampleConv(nn.Module):
    def __init__(self, in_chs, out_chs):
        super(UpsampleConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(in_chs, out_chs, kernel_size=3, stride=1, padding=1, bias=True),
            BatchNorm3d(out_chs, momentum=BN_MOMENTUM),
            ActivationFunction(inplace=relu_inplace)
        )

    def forward(self, x):
        return self.up(x)

class DownsampleConv(nn.Module):
    def __init__(self, in_chs, out_chs):
        super(DownsampleConv, self).__init__()
        self.down = nn.Sequential(
            nn.Conv3d(in_chs, out_chs, kernel_size=3, stride=2, padding=1, bias=False),
            BatchNorm3d(out_chs, momentum=BN_MOMENTUM),
            ActivationFunction(inplace=relu_inplace)
        )

    def forward(self, x):
        return self.down(x)

class SameSizeConv(nn.Module):
    def __init__(self, in_chs, out_chs):
        super(SameSizeConv, self).__init__()
        self.same = nn.Sequential(
            nn.Conv3d(in_chs, out_chs, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm3d(out_chs, momentum=BN_MOMENTUM),
            ActivationFunction(inplace=relu_inplace)
        )

    def forward(self, x):
        return self.same(x)

class TransitionConv(nn.Module):
    def __init__(self, in_chs, out_chs):
        super(TransitionConv, self).__init__()
        self.transition = nn.Sequential(
            nn.Conv3d(in_chs, out_chs, kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm3d(out_chs, momentum=BN_MOMENTUM),
            ActivationFunction(inplace=relu_inplace)
        )

    def forward(self, x):
        return self.transition(x)


class SideTransition(nn.Module):
    def __init__(self):
        super(SideTransition, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(32, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm3d(16, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm3d(8, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm3d(4, momentum=0.1),
            nn.ReLU(inplace=True))
    def forward(self, x):
        x = self.up(x)
        return x

class CustomBlock(nn.Module):
    expansion = 1

    def __init__(self, in_chs, out_chs, stride=1, downsample=None, norm_layer=BatchNorm3d):
        super(CustomBlock, self).__init__()
        self.conv1 = conv3x3(in_chs, out_chs, stride)
        self.bn1 = norm_layer(out_chs, momentum=BN_MOMENTUM)
        self.relu = ActivationFunction(inplace=relu_inplace)
        self.conv2 = conv3x3(out_chs, out_chs)
        self.bn2 = norm_layer(out_chs, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
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
    if not keepdim:
        temp_x = temp_x.squeeze()
    return temp_x

class HighFreqConv(nn.Module):
    def __init__(self, in_chs, out_chs, target_kernel):
        super(HighFreqConv, self).__init__()
        self.conv = nn.Conv3d(in_chs, out_chs, kernel_size=3, padding=1, bias=False)
        self.init_weights(target_kernel)

    def init_weights(self, target_kernel):
        with torch.no_grad():
            self.conv.weight = nn.Parameter(
                target_kernel.repeat(self.conv.out_channels, self.conv.in_channels, 1, 1, 1))

    def forward(self, x):
        return x + self.conv(x)

class SemanticAttentionModule(nn.Module):
    def __init__(self, in_features, reduction_rate=16):
        super(SemanticAttentionModule, self).__init__()
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
        super(PositionalAttentionModule, self).__init__()
        self.conv = nn.Conv3d(in_channels=2, out_channels=1, kernel_size=3, padding=1)  # Adjusted for 3D

    def forward(self, x):
        max_x = custom_max(x, dim=(1, 2), keepdim=True)  # Adjusted for 3D
        avg_x = torch.mean(x, dim=(1, 2), keepdim=True)  # Adjusted for 3D
        att = torch.cat((max_x, avg_x), dim=1)
        att = self.conv(att)
        att = torch.sigmoid(att)
        return x * att

class SliceAttentionModule(nn.Module):
    def __init__(self, in_features, rate=4, uncertainty=True, rank=5):
        super(SliceAttentionModule, self).__init__()
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
        # x: (B, C, D, H, W)
        B, C, D, H, W = x.shape

        # produce per-slice features (reduce over channel, height and width)
        # max over channel -> (B, D, H, W)
        max_x = x.max(dim=1)[0]
        # max over H -> (B, D, W)
        max_x = max_x.max(dim=2)[0]
        # max over W -> (B, D)
        max_x = max_x.max(dim=2)[0]

        # average over channel, height, width -> (B, D)
        avg_x = x.mean(dim=(1, 3, 4))  # (B, D)

        max_x = self.linear(max_x)    # (B, D)
        avg_x = self.linear(avg_x)    # (B, D)

        att = max_x + avg_x           # (B, D)

        if self.uncertainty:
            temp = self.non_linear(att)           # (B, D)
            mean = self.mean(temp)                # (B, D)
            # produce a positive diag and avoid tiny values
            diag = F.softplus(self.log_diag(temp)) + 1e-3   # (B, D)
            # produce factor and reshape -> (B, D, rank)
            factor = self.factor(temp).view(B, D, self.rank)

            # guard against NaN/Inf
            mean = torch.nan_to_num(mean, nan=0.0, posinf=1e6, neginf=-1e6)
            diag = torch.nan_to_num(diag, nan=1e-3, posinf=1e6, neginf=-1e6).clamp(min=1e-6)
            factor = torch.nan_to_num(factor, nan=0.0, posinf=1e6, neginf=-1e6)

            # try building the low-rank MVN, with jitter fallback
            try:
                dist = td.LowRankMultivariateNormal(loc=mean, cov_factor=factor, cov_diag=diag)
                samp = dist.rsample()  # or .sample() depending on needs
            except RuntimeError:
                # cholesky failed -> add increasing jitter to diag and retry
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
                    # final fallback: diagonal gaussian (stable)
                    eps = torch.randn_like(mean)
                    samp = mean + torch.sqrt(diag) * eps

            att = torch.sigmoid(samp).unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # (B,1,D,1,1)
        else:
            att = torch.sigmoid(att).unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # (B,1,D,1,1)

        # broadcast multiply: x * att  -> (B, C, D, H, W) * (B,1,D,1,1)
        return x * att

class FrequencyCrossAttention(nn.Module):
    def __init__(self, num_channels, num_slices , uncertainty=True, rank=5):
        super(FrequencyCrossAttention, self).__init__()
        self.semantic_att = SemanticAttentionModule(num_channels)
        self.positional_att = PositionalAttentionModule()
        self.slice_att = SliceAttentionModule(in_features=num_slices, uncertainty=uncertainty, rank=rank)

    def forward(self, x):
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

laplacian_3x3 = torch.tensor( [
            [[0, 0, 0],
             [0, 1, 0],
             [0, 0, 0]],

            [[0, 1, 0],
             [1, -6,1],
             [0, 1, 0]],

            [[0, 0, 0],
             [0, 1, 0],
             [0, 0, 0]]]
        , dtype=torch.float32)
upsampled_laplacian=laplacian_3x3

# --- MobileNet 3D helpers & small encoder ---
class HSwish(nn.Module):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0

class SELayer3D(nn.Module):
    def __init__(self, in_ch, reduction=4):
        super().__init__()
        mid = max(1, in_ch // reduction)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(in_ch, mid, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid, in_ch, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        s = self.pool(x)
        s = self.fc(s)
        return x * s

class InvertedResidual3D(nn.Module):
    def __init__(self, inp, hidden, outp, kernel, stride, use_se, nl='RE'):
        super().__init__()
        self.use_res_connect = (stride == 1 and inp == outp)
        Activation = HSwish if nl == 'HS' else nn.ReLU
        self.block = nn.Sequential(
            nn.Conv3d(inp, hidden, kernel_size=1, bias=False),
            BatchNorm3d(hidden, momentum=BN_MOMENTUM),
            Activation(inplace=relu_inplace),
            nn.Conv3d(hidden, hidden, kernel_size=kernel, stride=stride, padding=kernel//2, groups=hidden, bias=False),
            BatchNorm3d(hidden, momentum=BN_MOMENTUM),
            Activation(inplace=relu_inplace),
            SELayer3D(hidden) if use_se else nn.Identity(),
            nn.Conv3d(hidden, outp, kernel_size=1, bias=False),
            BatchNorm3d(outp, momentum=BN_MOMENTUM)
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_res_connect:
            return x + out
        return out

class MobileNetV3Small3D(nn.Module):
    def __init__(self, in_ch=16, layer_chs=(16,32,64,128,256)):
        super().__init__()
        l1, l2, l3, l4, l5 = layer_chs
        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, l1, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm3d(l1, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace)
        )
        self.l2 = nn.Sequential(
            InvertedResidual3D(l1, l1*2, l2, kernel=3, stride=2, use_se=True, nl='RE'),
            InvertedResidual3D(l2, l2*2, l2, kernel=3, stride=1, use_se=False, nl='RE')
        )
        self.l3 = nn.Sequential(
            InvertedResidual3D(l2, l2*2, l3, kernel=3, stride=2, use_se=True, nl='HS'),
            InvertedResidual3D(l3, l3*2, l3, kernel=3, stride=1, use_se=False, nl='HS')
        )
        self.l4 = nn.Sequential(
            InvertedResidual3D(l3, l3*2, l4, kernel=3, stride=2, use_se=True, nl='HS'),
            InvertedResidual3D(l4, l4*2, l4, kernel=3, stride=1, use_se=False, nl='HS')
        )
        self.l5 = nn.Sequential(
            InvertedResidual3D(l4, l4*2, l5, kernel=3, stride=2, use_se=True, nl='HS'),
            InvertedResidual3D(l5, l5*2, l5, kernel=3, stride=1, use_se=False, nl='HS')
        )

    def forward(self, x):
        f1 = self.stem(x)
        f2 = self.l2(f1)
        f3 = self.l3(f2)
        f4 = self.l4(f3)
        f5 = self.l5(f4)
        return f1, f2, f3, f4, f5


class HFFNet(nn.Module):
    def __init__(self, in_chs1,in_chs2,num_classes):
        super(HFFNet, self).__init__()

        layer1_chs,layer2_chs,layer3_chs,layer4_chs,layer5_chs =16, 32, 64, 128, 256
        self.laplacian_target = nn.Parameter(upsampled_laplacian.repeat(16, 16, 1, 1, 1), requires_grad=False)
        self.laplacian = upsampled_laplacian

        self.input_ed = HighFreqConv(16, 16, target_kernel=self.laplacian)
        self.LF_l4_FDCA = FrequencyCrossAttention(128, 16)
        self.LF_l5_FDCA= FrequencyCrossAttention(256, 8)
        self.HF_l4_FDCA = FrequencyCrossAttention(128, 16)
        self.HF_l5_FDCA = FrequencyCrossAttention(256, 8)

        self.x1_3side = SideTransition()
        self.x2_3side = SideTransition()

        # branch1
        self.l1_b1_1 = nn.Sequential(
            conv3x3(in_chs1, layer1_chs),
            conv3x3(layer1_chs, layer1_chs),
            CustomBlock(layer1_chs, layer1_chs)
        )
        self.l1_b1_d = DownsampleConv(layer1_chs, layer2_chs)
        self.l1_b1_2 = CustomBlock(layer1_chs + layer1_chs, layer1_chs, downsample=nn.Sequential(
            conv1x1(in_chs=layer1_chs + layer1_chs, out_chs=layer1_chs),
            BatchNorm3d(layer1_chs, momentum=BN_MOMENTUM)))
        self.l1_b1_f = nn.Conv3d(layer1_chs, num_classes, kernel_size=1, stride=1, padding=0)

        # branch1_layer2
        self.l2_b1_1 = CustomBlock(layer2_chs, layer2_chs)
        self.l2_b1_d = DownsampleConv(layer2_chs, layer3_chs)
        self.l2_b1_2 = CustomBlock(layer2_chs + layer2_chs, layer2_chs, downsample=nn.Sequential(
            conv1x1(in_chs=layer2_chs + layer2_chs, out_chs=layer2_chs),
            BatchNorm3d(layer2_chs, momentum=BN_MOMENTUM)))
        self.l2_b1_u = UpsampleConv(layer2_chs, layer1_chs)

        # branch1_layer3
        self.l3_b1_1 = CustomBlock(layer3_chs, layer3_chs)
        self.l3_b1_d = DownsampleConv(layer3_chs, layer4_chs)
        self.l3_b1_2 = CustomBlock(layer3_chs + layer3_chs, layer3_chs, downsample=nn.Sequential(
            conv1x1(in_chs=layer3_chs + layer3_chs, out_chs=layer3_chs),
            BatchNorm3d(layer3_chs, momentum=BN_MOMENTUM)))
        self.l3_b1_u = UpsampleConv(layer3_chs, layer2_chs)

        # branch1_layer4
        self.l4_b1_1 = CustomBlock(layer4_chs, layer4_chs)
        self.l4_b1_d = DownsampleConv(layer4_chs, layer5_chs)
        self.l4_b1_2 = CustomBlock(layer4_chs, layer4_chs)
        self.l4_b1_d2 = DownsampleConv(layer4_chs, layer4_chs)
        self.l4_b1_s = SameSizeConv(layer4_chs, layer4_chs)
        self.l4_b1_t = TransitionConv(layer4_chs + layer5_chs + layer4_chs, layer4_chs)
        self.l4_b1_3 = CustomBlock(layer4_chs, layer4_chs)
        self.l4_b1_4 = CustomBlock(layer4_chs + layer4_chs, layer4_chs, downsample=nn.Sequential(
            conv1x1(in_chs=layer4_chs + layer4_chs, out_chs=layer4_chs),
            BatchNorm3d(layer4_chs, momentum=BN_MOMENTUM)))
        self.l4_b1_u = UpsampleConv(layer4_chs, layer3_chs)

        # branch1_layer5
        self.l5_b1_1 = CustomBlock(layer5_chs, layer5_chs)
        self.l5_b1_u = UpsampleConv(layer5_chs, layer5_chs)
        self.l5_b1_s = SameSizeConv(layer5_chs, layer5_chs)
        self.l5_b1_t = TransitionConv(layer5_chs + layer5_chs + layer4_chs, layer5_chs)
        self.l5_b1_2 = CustomBlock(layer5_chs, layer5_chs)
        self.l5_b1_u2 = UpsampleConv(layer5_chs, layer4_chs)

        # branch2 downstream blocks (kept) - ensure these exist for decoder/merging
        self.l1_b2_1 = nn.Sequential(
            conv3x3(in_chs2, layer1_chs),
            conv3x3(layer1_chs, layer1_chs),
            CustomBlock(layer1_chs, layer1_chs)
        )
        self.l1_b2_d = DownsampleConv(layer1_chs, layer2_chs)
        self.l1_b2_2 = CustomBlock(layer1_chs + layer1_chs, layer1_chs,
                                   downsample=nn.Sequential(conv1x1(layer1_chs + layer1_chs, layer1_chs),
                                                            BatchNorm3d(layer1_chs, momentum=BN_MOMENTUM)))
        self.l1_b2_f = nn.Conv3d(layer1_chs, num_classes, kernel_size=1, stride=1, padding=0)
        # layer2
        self.l2_b2_1 = CustomBlock(layer2_chs, layer2_chs)
        self.l2_b2_d = DownsampleConv(layer2_chs, layer3_chs)
        self.l2_b2_2 = CustomBlock(layer2_chs + layer2_chs, layer2_chs,
                                   downsample=nn.Sequential(conv1x1(layer2_chs + layer2_chs, layer2_chs),
                                                            BatchNorm3d(layer2_chs, momentum=BN_MOMENTUM)))
        self.l2_b2_u = UpsampleConv(layer2_chs, layer1_chs)
        # branch2_layer3
        self.l3_b2_1 = CustomBlock(layer3_chs, layer3_chs)
        self.l3_b2_d = DownsampleConv(layer3_chs, layer4_chs)
        self.l3_b2_2 = CustomBlock(layer3_chs + layer3_chs, layer3_chs,
                                   downsample=nn.Sequential(conv1x1(layer3_chs + layer3_chs, layer3_chs),
                                                            BatchNorm3d(layer3_chs, momentum=BN_MOMENTUM)))
        self.l3_b2_u = UpsampleConv(layer3_chs, layer2_chs)
        # branch2_layer4
        self.l4_b2_1 = CustomBlock(layer4_chs, layer4_chs)
        self.l4_b2_d = DownsampleConv(layer4_chs, layer5_chs)
        self.l4_b2_2 = CustomBlock(layer4_chs, layer4_chs)
        self.l4_b2_d2 = DownsampleConv(layer4_chs, layer4_chs)
        self.l4_b2_s = SameSizeConv(layer4_chs, layer4_chs)
        self.l4_b2_t = TransitionConv(layer4_chs + layer5_chs + layer4_chs, layer4_chs)
        self.l4_b2_3 = CustomBlock(layer4_chs, layer4_chs)
        self.l4_b2_4 = CustomBlock(layer4_chs + layer4_chs, layer4_chs,
                                   downsample=nn.Sequential(conv1x1(layer4_chs + layer4_chs, layer4_chs),
                                                            BatchNorm3d(layer4_chs, momentum=BN_MOMENTUM)))
        self.l4_b2_u = UpsampleConv(layer4_chs, layer3_chs)
        # branch2_layer5
        self.l5_b2_1 = CustomBlock(layer5_chs, layer5_chs)
        self.l5_b2_u = UpsampleConv(layer5_chs, layer5_chs)
        self.l5_b2_s = SameSizeConv(layer5_chs, layer5_chs)
        self.l5_b2_t = TransitionConv(layer5_chs + layer5_chs + layer4_chs, layer5_chs)
        self.l5_b2_2 = CustomBlock(layer5_chs, layer5_chs)
        self.l5_b2_u2 = UpsampleConv(layer5_chs, layer4_chs)

        # MobileNetV3 small 3D encoder for HF branch (lightweight encoder)
        self.mobilenet3d = MobileNetV3Small3D(in_ch=in_chs2, layer_chs=(layer1_chs, layer2_chs, layer3_chs, layer4_chs, layer5_chs))

        for m in self.modules():
            if isinstance(m, HighFreqConv):
                continue
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input1, input2):
        # branch1 (LF)
        layer1_LF = self.l1_b1_1(input1)
        layer2_LF = self.l1_b1_d(layer1_LF)
        layer2_LF = self.l2_b1_1(layer2_LF)

        layer3_LF = self.l2_b1_d(layer2_LF)
        layer3_LF = self.l3_b1_1(layer3_LF)

        layer4_LF_1 = self.l3_b1_d(layer3_LF)
        layer4_LF_1 = self.l4_b1_1(layer4_LF_1)
        layer4_LF_2 = self.l4_b1_2(layer4_LF_1)
        layer4_LF_3_d = self.l4_b1_d2(layer4_LF_2)
        layer4_LF_3_s = self.l4_b1_s(layer4_LF_2)

        layer4_LF_3_s = self.LF_l4_FDCA(layer4_LF_3_s)

        layer5_LF_1 = self.l4_b1_d(layer4_LF_1)
        layer5_LF_1 = self.l5_b1_1(layer5_LF_1)
        layer5_LF_u = self.l5_b1_u(layer5_LF_1)
        layer5_LF_s = self.l5_b1_s(layer5_LF_1)
        layer5_LF_s = self.LF_l5_FDCA(layer5_LF_s)

        # branch2 (HF) — use MobileNet3D encoder + existing downstream blocks
        input2 = self.input_ed(input2)  # high-frequency preconv
        # get multi-scale features from MobileNet encoder
        layer1_HF, layer2_HF, layer3_HF, layer4_HF_1, layer5_HF_1 = self.mobilenet3d(input2)

        # downstream HF processing (use existing blocks)
        layer4_HF_1 = self.l4_b2_1(layer4_HF_1)
        layer4_HF_2 = self.l4_b2_2(layer4_HF_1)
        layer4_HF_3_d = self.l4_b2_d2(layer4_HF_2)
        layer4_HF_3_s = self.l4_b2_s(layer4_HF_2)
        layer4_HF_3_s = self.HF_l4_FDCA(layer4_HF_3_s)

        layer5_HF_1 = self.l5_b2_1(layer5_HF_1)
        layer5_HF_u = self.l5_b2_u(layer5_HF_1)
        layer5_HF_s = self.l5_b2_s(layer5_HF_1)
        layer5_HF_s = self.HF_l5_FDCA(layer5_HF_s)

        # merge
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

        # decode
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


def hff_net(in_chs1,in_chs2, num_classes):
    return HFFNet(in_chs1,in_chs2, num_classes)


if __name__ == "__main__":
    model = hff_net(4,16, 4).cuda()
    input1 = torch.rand(1, 4, 128, 128, 128).cuda()
    input2 = torch.rand(1, 16, 128, 128, 128).cuda()
    x1_1_main, x1_1_aux1, x2, x3 = model(input1, input2)
    print(x2.shape)
