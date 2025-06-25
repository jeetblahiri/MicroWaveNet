import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft, rfftfreq
import torch
from torch.nn import Module
from torch import nn
from prettytable import PrettyTable
import ptwt
import pywt
from tslearn.metrics import SoftDTWLossPyTorch

class LayerNorm(nn.Module):
    def __init__(self, num_features):
        super(LayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(num_features)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.layer_norm(x)
        x = x.permute(0, 2, 1)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
           
        self.fc = nn.Sequential(nn.Conv1d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv1d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_channels, ratio=16, kernel_size=7, residual=False):
        super(CBAM, self).__init__()

        # basic CBAM block.
        self.channel_attention = ChannelAttention(in_channels, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        self.relu = nn.ReLU()
        self.residual = residual
    
    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out

        if self.residual:
            out += x
        
        return out

class Encoder(Module):
    def __init__(self, wavelet=''):
        super(Encoder, self).__init__()

        self.wavelet = wavelet
        self.eeg_skip_states = []
        self.art_skip_states = []
        self.skip_states = []
        self.eeg_cbam_skip_states = []
        self.art_cbam_skip_states = []

        self.pre_wavelet = nn.Sequential(
            # input size - 1x512, output size - 12x512
            # 4 sets of 3 freq components
            # can be obtained by averaging 4 sub-regions of freq region

            # first split into 3 streams
            nn.Conv1d(1, 3, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.BatchNorm1d(3),
            nn.LeakyReLU()
        )
        self.wavelet_1_stream = nn.Sequential(
            # input - 1x512, output - 4x512
            nn.Conv1d(1, 2, kernel_size=21, stride=1, padding=10, padding_mode='reflect'),
            nn.BatchNorm1d(2),
            nn.LeakyReLU(),
            nn.Conv1d(2, 4, kernel_size=9, stride=1, padding=4, padding_mode='reflect'),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(),
        )

        self.wavelet_2_stream = nn.Sequential(
            # input - 1x512, output - 4x512
            nn.Conv1d(1, 2, kernel_size=11, stride=1, padding=5, padding_mode='reflect'),
            nn.BatchNorm1d(2),
            nn.LeakyReLU(),
            nn.Conv1d(2, 4, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(),
        )

        self.wavelet_3_stream = nn.Sequential(
            # input - 1x512, output - 4x512
            nn.Conv1d(1, 2, kernel_size=5, stride=1, padding=2, padding_mode='reflect'),
            nn.BatchNorm1d(2),
            nn.LeakyReLU(),
            nn.Conv1d(2, 4, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(),
        )

        self.wavelet_1_stream_post_loss = nn.Sequential(
            # input - 4x512, output - 8x64
            nn.Conv1d(4, 8, kernel_size=15, stride=2, padding=7, padding_mode='reflect'),
            nn.BatchNorm1d(8),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.Conv1d(8, 8, kernel_size=9, stride=1, padding=4, padding_mode='reflect'),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.1)
        )
        
        self.wavelet_2_stream_post_loss = nn.Sequential(
            # input - 4x512, output - 8x64
            nn.Conv1d(4, 8, kernel_size=9, stride=2, padding=4, padding_mode='reflect'),
            nn.BatchNorm1d(8),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.Conv1d(8, 8, kernel_size=5, stride=1, padding=2, padding_mode='reflect'),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.1)
        )

        self.wavelet_3_stream_post_loss = nn.Sequential(
            # input - 4x512, output - 8x64
            nn.Conv1d(4, 8, kernel_size=5, stride=2, padding=2, padding_mode='reflect'),
            nn.BatchNorm1d(8),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.Conv1d(8, 8, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.1)
        )

        self.pre_latent0_CBAM = nn.Sequential(
            CBAM(24, ratio=8),
            LayerNorm(24),
            nn.LeakyReLU()
        )

        self.pre_latent0 = nn.Sequential(
            # in - 24x64, out - 32x64
            nn.Conv1d(24, 32, kernel_size=5, stride=1, padding=2, padding_mode='reflect'),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Dropout(0.1)
        )

        # change pre-latent to increase depth while decreasing length
        # change LP and HP final depths
        # can later increase to 2 convs per depth level

        self.pre_latent1_eeg_CBAM = nn.Sequential(
            CBAM(32, ratio=16),
            LayerNorm(32),
            nn.LeakyReLU()
        )

        self.pre_latent1_eeg = nn.Sequential(
            # do some conv and then bring to two channels
            # can do dws to reduce params
            # in - 32x64, out - 64x32
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2, padding_mode='reflect', groups=8),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.15)
        )

        self.pre_latent1_artefact_CBAM = nn.Sequential(
            CBAM(32, ratio=16),
            LayerNorm(32),
            nn.LeakyReLU()
        )

        self.pre_latent1_artefact = nn.Sequential(
            # in - 32x64, out - 64x32
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2, padding_mode='reflect', groups=8),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.15)
        )

        self.pre_latent2_eeg_CBAM = nn.Sequential(
            CBAM(64, ratio=32),
            LayerNorm(64),
            nn.LeakyReLU()
        )

        self.pre_latent2_eeg = nn.Sequential(
            # in - 64x32, out - 128x16
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2, padding_mode='reflect', groups=16),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.25)
        )

        self.pre_latent2_artefact_CBAM = nn.Sequential(
            CBAM(64, ratio=32),
            LayerNorm(64),
            nn.LeakyReLU()
        )

        self.pre_latent2_artefact = nn.Sequential(
            # in - 64x32, out - 128x16
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2, padding_mode='reflect', groups=32),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.25)
        )

        self.pre_latent3_eeg_CBAM = nn.Sequential(
            CBAM(128, ratio=64),
            LayerNorm(128),
            nn.LeakyReLU()
        )

        self.pre_latent3_eeg = nn.Sequential(
            # in - 128x16, out - 128x8
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Dropout(0.25)
        )

        self.pre_latent3_artefact_CBAM =  nn.Sequential(
            CBAM(128, ratio=64),
            LayerNorm(128),
            nn.LeakyReLU()
        )

        self.pre_latent3_artefact = nn.Sequential(
            # in - 128x16, out - 128x8
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Dropout(0.25)
        )
        


    def standardize(self, x):
        # z-score
        return (x - x.mean())/x.std()


    def forward(self, x):
        self.skip_states = []
        self.eeg_skip_states = []
        self.art_skip_states = []
        self.eeg_cbam_skip_states = []
        self.art_cbam_skip_states = []

        x = self.pre_wavelet(x)
        #print(f'pre-wavelet shape: {x.shape}')
        arr = torch.split(x, 1, dim=1)

        l = self.wavelet_1_stream(arr[0])
        m = self.wavelet_2_stream(arr[1])
        h = self.wavelet_3_stream(arr[2])
        #print(f'l, m, h shape: {l.shape}, {m.shape}, {h.shape}')

        wavelet_loss_tensor = torch.concat([l, m, h], dim=1)
        #print(f'wavelet loss tensor shape (skip state 1): {wavelet_loss_tensor.shape}')
        self.skip_states.append(wavelet_loss_tensor)

        l = self.wavelet_1_stream_post_loss(l)
        m = self.wavelet_2_stream_post_loss(m)
        h = self.wavelet_3_stream_post_loss(h)
        #print(f'l, m, h post loss shape: {l.shape}, {m.shape}, {h.shape}')
        
        wavelet_post_loss_tensor = torch.concat([l, m, h], dim=1)
        self.skip_states.append(wavelet_post_loss_tensor)
        #print(f'wavelet post loss tensor shape (skip state 2): {wavelet_post_loss_tensor.shape}')

        x = torch.concat([l, m, h], dim=1)
        x = self.pre_latent0_CBAM(x)
        self.art_cbam_skip_states.append(x.clone())
        self.eeg_cbam_skip_states.append(x.clone())
        #print(f'pre-latent0 CBAM shape (CBAM skip state 1): {x.shape}')

        x = self.pre_latent0(x)
        #print(f'pre-latent0 shape (skip state 3): {x.shape}')
        self.skip_states.append(x.clone())

        # pre-latent, split into eeg embedding and artefact embedding
        x_eeg = self.pre_latent1_eeg_CBAM(x)
        x_artefact = self.pre_latent1_artefact_CBAM(x)
        self.eeg_cbam_skip_states.append(x_eeg.clone())
        self.art_cbam_skip_states.append(x_artefact.clone())
        #print(f'pre-latent1 CBAM shape (CBAM skip state 2): {x_eeg.shape}')

        x_eeg = self.pre_latent1_eeg(x)
        x_artefact = self.pre_latent1_artefact(x)
        self.eeg_skip_states.append(x_eeg.clone())
        self.art_skip_states.append(x_artefact.clone())
        #print(f'pre-latent1 shape (eeg skip state 1): {x_eeg.shape}')
        
        x_eeg = self.pre_latent2_eeg_CBAM(x_eeg)
        x_artefact = self.pre_latent2_artefact_CBAM(x_artefact)
        self.eeg_cbam_skip_states.append(x_eeg.clone())
        self.art_cbam_skip_states.append(x_artefact.clone())
        #print(f'pre-latent2 CBAM shape (CBAM skip state 3): {x_eeg.shape}')

        x_eeg = self.pre_latent2_eeg(x_eeg)
        x_artefact = self.pre_latent2_artefact(x_artefact)
        self.eeg_skip_states.append(x_eeg.clone())
        self.art_skip_states.append(x_artefact.clone())
        #print(f'pre-latent2 shape (eeg skip state 2): {x_eeg.shape}')

        x_eeg = self.pre_latent3_eeg_CBAM(x_eeg)
        x_artefact = self.pre_latent3_artefact_CBAM(x_artefact)
        # self.eeg_cbam_skip_states.append(x_eeg.clone())
        # self.art_cbam_skip_states.append(x_artefact.clone())
        # print(f'pre-latent3 CBAM shape (CBAM skip state 4): {x_eeg.shape}')

        x_eeg = self.pre_latent3_eeg(x_eeg)
        x_artefact = self.pre_latent3_artefact(x_artefact)

        #print(f'pre-latent3 shape: {x_eeg.shape}')
        #print(f'pre-latent3 eeg shape: {x_eeg.shape}, artefact shape: {x_artefact.shape}')
        
        # print(f'eeg shape: {eeg.shape}, artefact shape: {artefact.shape}')
        return x_eeg, x_artefact, wavelet_loss_tensor
    
class Decoder(Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.post_latent3 = nn.Sequential(
            # in - 128x8, out - 128x16
            nn.ConvTranspose1d(128, 128, kernel_size=4, stride=2, padding=1, groups=16),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.25)
        )

        self.post_latent3_CBAM = nn.Sequential(
            CBAM(128, ratio=64),
            LayerNorm(128),
            nn.LeakyReLU()
        )

        self.post_latent2 = nn.Sequential(
            # in - 256x16 (after cat), out - 64x32
            nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1, padding_mode='reflect', groups=64),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=6, stride=2, padding=2, groups=32),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.25)
        )

        self.post_latent2_CBAM = nn.Sequential(
            # in 128x32 (after cat), out - 64x32
            nn.Conv1d(128, 64, kernel_size=1, stride=1, padding=0, groups=32),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            CBAM(64, ratio=32),
            LayerNorm(64),
            nn.LeakyReLU()
        )

        self.post_latent1 = nn.Sequential(
            # in - 128x32 (after cat), out - 32x64
            nn.Conv1d(128, 64, kernel_size=5, stride=1, padding=2, padding_mode='reflect', groups=32),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=6, stride=2, padding=2, groups=16),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Dropout(0.15)
        )

        self.post_latent1_CBAM = nn.Sequential(
            # in - 64x64 (after cat), out - 32x64
            nn.Conv1d(64, 32, kernel_size=1, stride=1, padding=0, groups=16),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            CBAM(32, ratio=16),
            LayerNorm(32),
            nn.LeakyReLU()
        )

        self.post_latent0 = nn.Sequential(
            # in - 64x64 (after cat), out - 24x64
            nn.Conv1d(64, 32, kernel_size=5, stride=1, padding=2, padding_mode='reflect', groups=16),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Conv1d(32, 24, kernel_size=5, stride=1, padding=2, groups=8),
            nn.BatchNorm1d(24),
            nn.LeakyReLU(),
            nn.Dropout(0.1)
        )

        self.post_latent0_CBAM = nn.Sequential(
            # in - 48x64 (after cat), out - 24x64
            nn.Conv1d(48, 24, kernel_size=1, stride=1, padding=0, groups=8),
            nn.BatchNorm1d(24),
            nn.LeakyReLU(),
            CBAM(24, ratio=8),
            LayerNorm(24),
            nn.LeakyReLU()
        )

        # split into 3 8x64 streams
        self.wavelet_1_stream_pre_loss = nn.Sequential(
            # in - 16x64 (after concat), out - 4x512
            nn.ConvTranspose1d(16, 8, kernel_size=6, stride=2, padding=2),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(8, 4, kernel_size=14, stride=4, padding=5),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(),
            nn.Dropout(0.1)
        )

        self.wavelet_2_stream_pre_loss = nn.Sequential(
            # in - 16x64, out - 4x512
            nn.ConvTranspose1d(16, 8, kernel_size=6, stride=2, padding=2),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(8, 4, kernel_size=10, stride=4, padding=3),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(),
            nn.Dropout(0.1)
        )

        self.wavelet_3_stream_pre_loss = nn.Sequential(
            # in - 16x64, out - 4x512
            nn.ConvTranspose1d(16, 8, kernel_size=6, stride=2, padding=2),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(8, 4, kernel_size=6, stride=4, padding=1),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(),
            nn.Dropout(0.1)
        )

        self.wavelet_1_stream_post_loss = nn.Sequential(
            # in - 8x512 (after cat), out - 1x512
            nn.ConvTranspose1d(8, 2, kernel_size=21, stride=1, padding=10),
            nn.BatchNorm1d(2),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(2, 1, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(1),
            nn.LeakyReLU()
        )
        self.wavelet_2_stream_post_loss = nn.Sequential(
            # in - 8x512 (after cat), out - 1x512
            nn.ConvTranspose1d(8, 2, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm1d(2),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(2, 1, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(1),
            nn.LeakyReLU()
        )
        self.wavelet_3_stream_post_loss = nn.Sequential(
            # in - 8x512 (after cat), out - 1x512
            nn.ConvTranspose1d(8, 2, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(2),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(2, 1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(1),
            nn.LeakyReLU()
        )

        # merge low, mid, high freq streams
        self.post_wavelet = nn.Sequential(
            # in - 3x512, out - 1x512
            nn.ConvTranspose1d(3, 1, kernel_size=3, stride=1, padding=1),
        )

        self.window_weights = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=16, stride=1, padding=8),
            nn.AvgPool1d(kernel_size=16, stride=16)
        )



    def forward(self, x, skip_states, eegart_skip_states, cbam_skip_states, origsig):
        # pre-latent
        x = self.post_latent3(x)
        x = self.post_latent3_CBAM(x)
        #print(f'post-latent3 CBAM shape: {x.shape}')
        #print(f'post-latent3 shape: {x.shape}')
        
        x = torch.cat([x, eegart_skip_states[-1]], dim=1)
        x = self.post_latent2(x)
        #print(f'post-latent2 shape: {x.shape}')
        
        x = torch.cat([x, cbam_skip_states[-1]], dim=1)
        x = self.post_latent2_CBAM(x)

        #print(f'post-latent2 shape: {x.shape}')
        x = torch.cat([x, eegart_skip_states[-2]], dim=1)
        x = self.post_latent1(x)
        
        x = torch.cat([x, cbam_skip_states[-2]], dim=1)
        x = self.post_latent1_CBAM(x)
        
        #print(f'post-latent1 shape: {x.shape}')
        x = torch.cat([x, skip_states[-1]], dim=1)
        x = self.post_latent0(x)
        
        x = torch.cat([x, cbam_skip_states[-3]], dim=1)
        x = self.post_latent0_CBAM(x)
        
        #print(f'post-latent0 shape: {x.shape}')
        arr = torch.split(x, 8, dim=1)
        #print(f'after splitting shape : {arr[0].shape}, {arr[1].shape}, {arr[2].shape}')

        skip_arr = torch.split(skip_states[-2], 8, dim=1)
        l = torch.concat([arr[0], skip_arr[0]], dim=1)
        m = torch.concat([arr[1], skip_arr[1]], dim=1)
        h = torch.concat([arr[2], skip_arr[2]], dim=1)

        l = self.wavelet_1_stream_pre_loss(l)
        m = self.wavelet_2_stream_pre_loss(m)
        h = self.wavelet_3_stream_pre_loss(h)
        #print(f'l, m, h pre loss shape: {l.shape}, {m.shape}, {h.shape}')

        skip_arr = torch.split(skip_states[-3], 4, dim=1)
        l = torch.concat([l, skip_arr[0]], dim=1)
        m = torch.concat([m, skip_arr[1]], dim=1)
        h = torch.concat([h, skip_arr[2]], dim=1)

        l = self.wavelet_1_stream_post_loss(l)
        m = self.wavelet_2_stream_post_loss(m)
        h = self.wavelet_3_stream_post_loss(h)
        #print(f'l, m, h post loss shape: {l.shape}, {m.shape}, {h.shape}')

        x = self.post_wavelet(torch.concat([l, m, h], dim=1))
        #print(f'post-wavelet shape: {x.shape}')

        diff = origsig - x
        weights = self.window_weights(diff)

        # x = x.view(-1, 512/16, 16)
        B, _, N = x.shape
        x = x.view(B, 1, -1, 16)                  # [B, 1, N//16, 16]
        x = x * weights[..., None]               # broadcast to [B, 1, N//16, 16]
        x = x.view(B, 1, N)                      # back to [B, 1, N]
        return x
    

class EEGNetMorletWindowCBAMDropout(Module):
    def __init__(self,
                 device,
                 wavelet='cmor1.0-1.5', 
                 dt=1/256,
                 lowfreq=0.5,
                 highfreq=30,
                 nfreq = 48,
                 gamma=0.1, 
                 waveletWeight=0.1
                 ):
        super(EEGNetMorletWindowCBAMDropout, self).__init__()
        self.encoder = Encoder()
        self.eeg_decoder = Decoder()
        self.art_decoder = Decoder()
        
        self.wavelet = wavelet
        self.dt = dt
        fc = pywt.central_frequency(wavelet)
        self.freqs = np.linspace(lowfreq, highfreq, nfreq)
        self.scales = fc / (self.freqs * dt)

        self.gamma = gamma
        self.w = waveletWeight

        self.log_vars = nn.Parameter(torch.zeros(3))

        self.dtwLoss = SoftDTWLossPyTorch(gamma=self.gamma, normalize=True)
        self.mseLoss = nn.MSELoss(reduction='mean')
        self.maeLoss = nn.L1Loss(reduction='mean')

        self.device=device

    def standardize(self, x):
        # z-score
        return (x - x.mean())/x.std()

    def forward(self, x):
        # verify that skip states are being passed correctly.
        x_inp = x.clone()
        eeg_z, artefact_z, wav_tensor = self.encoder(x)
        # print(len(self.encoder.skip_states))
        eeg = self.eeg_decoder(eeg_z, self.encoder.skip_states, self.encoder.eeg_skip_states, self.encoder.eeg_cbam_skip_states, x_inp)
        artefact = self.art_decoder(artefact_z, self.encoder.skip_states, self.encoder.art_skip_states, self.encoder.art_cbam_skip_states, x_inp)

        return eeg, artefact, eeg_z, artefact_z, wav_tensor, x_inp

    def reconstructionLoss(self, x, y):
        loss_dtw = self.dtwLoss(x, y).mean()
        loss_mse = self.mseLoss(x, y)
        loss_mae = self.maeLoss(x, y)

        losses = torch.stack([loss_dtw, loss_mse, loss_mae])
    
        total_loss = torch.sum(torch.exp(-2 * self.log_vars) * losses + self.log_vars)
        return total_loss
        # return loss_mse

    def mutualInfoLoss(self):
        # multiply by a small scaling factor, because MI loss will encourage x = constant
        return 0

    def waveletLoss(self, x, wav):
        return self.dtwLoss(x, wav).mean()

    def get_wavelet_features(self, x):
        coeffs, _ = ptwt.cwt(x, self.scales, self.wavelet, sampling_period=self.dt)
        coeffs = torch.abs(coeffs).squeeze().transpose(0, 1)

        n_freqs = coeffs.shape[1]
        # lowidx = n_freqs // 3
        # mididx = 2 * n_freqs // 3

        lowcoeffs, midcoeffs, highcoeffs = torch.split(coeffs, n_freqs // 3, dim=1)

        tempcoeffs = []
        for part in [lowcoeffs, midcoeffs, highcoeffs]:
            B, F, T = part.shape
            part = part[:, :F - (F % 4), :]  # make divisible by 4
            part = part.reshape(B, 4, F // 4, T).mean(dim=2)
            tempcoeffs.append(part)

        wavelet_features = torch.cat(tempcoeffs, dim=1)
        return wavelet_features


    def loss(self, f, eeg_y, artefact_y):
        # time this, tuple unpacking can be slow
        eeg, artefact, eeg_z, artefact_z, wav_tensor, x_inp = f
        

        eegrec = self.reconstructionLoss(eeg, eeg_y)
        artefactrec = self.reconstructionLoss(artefact, artefact_y)

        mim = self.mutualInfoLoss()

        wavelet_y = self.get_wavelet_features(x_inp)
        wvl = self.waveletLoss(wav_tensor, wavelet_y)
        
        total = eegrec + artefactrec + mim + self.w*wvl
        return eegrec, artefactrec, mim, self.w*wvl, total

    def dwt(self, x, wavelet='db2'):
        # cA is low, cD is high
        # question - which signal extension mode to use?
        # question/idead - since LP is very close to original signal (maybe because we're using EOG)
        # we can try pyramid DWT and use second order LP and HP, or even all 3.
        # could define custom wavelet according to frequency bands preference

        cA, cD = ptwt.wavedec(x, wavelet, level=1)
        # print(f'inside dwt, x - {x.shape}, cA - {cA.shape}, cD - {cD.shape}')
        return cA, cD