import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import dataloader
from utils import count_parameters
from multiwaveletmodel import EEGNetMorlet
from dataset import get_data, EEGDataset, get_data_non_std, get_wavelet_features, EEGDatasetPCWavelet
from tqdm import tqdm
import argparse
from attention_model_sequential import EEGNet_Attention_Interleave
from base_model import EEGNet
from mwmwindow import EEGNetMorletWindow
from mwmwindow_alignloss import EEGNetMorletWindowAlignLoss
from cbam import EEGNetMorletWindowCBAM
from cbamdropout import EEGNetMorletWindowCBAMDropout

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(f'using device - {device}')

parser = argparse.ArgumentParser()
parser.add_argument('--data_cont', type=str)
parser.add_argument('--data_clean', type=str)
parser.add_argument('--test_frac', type=float)
parser.add_argument('--val_frac', type=float)
parser.add_argument('--bsize', type=int)
parser.add_argument('--filename', type=str)
parser.add_argument('--epochs', type=int)
parser.add_argument('--preload', type=int, default=0)
parser.add_argument('--preload_epoch', type=int, default=0)
parser.add_argument('-std', '--standardize', type=int, default=1)
parser.add_argument('--model', type=str, default='base')
parser.add_argument('--scheduler', type=str, default='onecycle')
parser.add_argument('--div', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--float', type=int, default=32)
parser.add_argument('--optim', type=str, default='adam')

args = parser.parse_args()
datapath_conts = args.data_cont
datapath_cleans = args.data_clean
test_frac = args.test_frac
val_frac = args.val_frac
bsize = args.bsize
filename = args.filename
epochs = args.epochs
preload = args.preload
preload_epoch = args.preload_epoch
standardize = args.standardize
model_name = args.model
schedname = args.scheduler
div = args.div
startlr = args.lr
float_prec = args.float
optim = args.optim

if float_prec == 32:
    torch.set_default_dtype(torch.float32)
    set_dtype = torch.float32
    print('using float32')
elif float_prec == 64:
    torch.set_default_dtype(torch.float64)
    set_dtype = torch.float64
    print('using float64')
else:
    raise NotImplementedError(f"Float precision {float_prec} not implemented/recognised!")

srate = 256
T = 2
t = np.linspace(0, T, srate*T)

if preload:
    modelpath = f"{filename}_epoch_{preload_epoch}.pt"
    model = torch.load(modelpath, weights_only=False, map_location=device)
else:
    if model_name == 'base':
        model = EEGNet()
        print('using base model')
    elif model_name == 'morlet':
        model = EEGNetMorlet(device=device)
        print('using morlet model')
    elif model_name == 'attention':
        model = EEGNet_Attention_Interleave()
        print('using attention model')
    elif model_name == 'morlet_window':
        model = EEGNetMorletWindow(device=device)
        print('using morlet window model')
    elif model_name == 'mwm_align':
        model = EEGNetMorletWindowAlignLoss(device=device)
        print('using morlet window align loss model')
    elif model_name == 'cbam':
        model = EEGNetMorletWindowCBAM(device=device)
        print('using morlet window cbam model')
    elif model_name == 'cbamdrop':
        model = EEGNetMorletWindowCBAMDropout(device=device)
        print('using morlet window cbam dropout model')
    else:
        raise NotImplementedError(f"Model {model_name} not implemented/recognised!")

model = model.to(device, dtype=set_dtype)

count_parameters(model)
print(f'preload - {preload}, preload_epoch - {preload_epoch}')

# load data
if standardize:
    print('standardizing data!')
    trainx, trainy, valx, valy, train_art_y, val_art_y = get_data(datapath_conts, datapath_cleans, val_frac, set_dtype, div)
else:
    print('not standardizing data!')
    trainx, trainy, valx, valy, train_art_y, val_art_y = get_data_non_std(datapath_conts, datapath_cleans, val_frac, set_dtype, div)

# valx = trainx[int(len(trainx)*(1-val_frac)):]
# valy = trainy[int(len(trainy)*(1-val_frac)):]
# trainx = trainx[:int(len(trainx)*(1-val_frac))]
# trainy = trainy[:int(len(trainy)*(1-val_frac))]

# trainwav = get_wavelet_features(trainx, model.scales, model.wavelet, model.dt)
# valwav = get_wavelet_features(valx, model.scales, model.wavelet, model.dt)

trainSet = EEGDataset(trainx, trainy, train_art_y)
valSet = EEGDataset(valx, valy, val_art_y)

trainDataLoader = dataloader.DataLoader(trainSet, batch_size=bsize, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
valDataLoader = dataloader.DataLoader(valSet, batch_size=bsize, shuffle=True, drop_last=True, pin_memory=True, num_workers=2)

# train loop

batches_per_epoch = len(trainDataLoader)

if optim == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=startlr)
elif optim == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=startlr, momentum=0.9, weight_decay=1e-4)
elif optim == 'rms':
    optimizer = torch.optim.RMSprop(model.parameters(), lr=startlr, alpha=0.9, eps=1e-08, weight_decay=0, momentum=0)

if preload_epoch >= 0:
    for group in optimizer.param_groups:
        group.setdefault('initial_lr', group['lr'])
        group.setdefault('max_lr', 5e-3)
        group.setdefault('min_lr', 1e-5)
        group.setdefault('base_momentum', 0)
        group.setdefault('max_momentum', 0.95)

if schedname == 'onecycle':
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=startlr,        
        total_steps=batches_per_epoch*(epochs+preload_epoch),  
        pct_start=0.3,    
        anneal_strategy='cos', 
        final_div_factor=1e3,
        last_epoch=batches_per_epoch*preload_epoch - 1
    )
elif schedname == 'plateau':
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=5,
        threshold=1e-4,
        threshold_mode='abs'
    )
elif schedname == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
    )
elif schedname == 'const':
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: 1.0
    )
else:
    raise NotImplementedError(f"Scheduler {schedname} not implemented/recognised!")

print(f'start lr is {startlr}')

trainLosses = []
valLosses = []
trainRecLosses = []
valRecLosses = []

lossfilename = f'{filename}_losses.txt'

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    pbar = tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), desc=f"Epoch {epoch+1}/{epochs}")

    # track individual losses
    eegloss = 0
    artefactloss = 0
    wvlloss = 0

    for j, data in pbar:
        x, y, art = data

        x = x.to(device)
        y = y.to(device)
        art = art.to(device)
        # wav = wav.to(device)

        # print(f'x: {x.shape}, y: {y.shape}, art: {art.shape}')

        optimizer.zero_grad()
        out = model(x)
        # print(f'eegrec shape: {out[0].shape}, artefactrec shape: {out[1].shape}, mim shape: {out[2].shape}, wvl shape: {out[3].shape}, loss shape: {out[4].shape}')
        eegrec, artefactrec, mim, wvl, loss = model.loss(out, y, art)

        loss.backward()
        optimizer.step()
        if schedname != 'plateau':
            scheduler.step()

        batch_loss = loss.item() / bsize
        epoch_loss += batch_loss

        # eegloss += eegrec.item() / bsize
        # artefactloss += artefactrec.item() / bsize
        # wvlloss += wvl.item() / bsize

        pbar.set_postfix(batch_loss=batch_loss)

    print(f'Epoch {epoch+1}, Training Loss: {epoch_loss/len(trainy)}')

    trainLosses.append(epoch_loss)
    with open(lossfilename, 'a') as f:
        f.write(f'{epoch+1}, {epoch_loss}\n')

    # val
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for i, data in enumerate(valDataLoader):
            x, y, art = data

            x = x.to(device)
            y = y.to(device)
            art = art.to(device)
            # wav = wav.to(device)

            out = model(x)
            eegrec, artefactrec, mim, wvl, loss = model.loss(out, y, art)
            batch_loss = loss.item() / bsize
            val_loss += batch_loss

            eegloss += eegrec.item() / bsize
            artefactloss += artefactrec.item() / bsize
            wvlloss += wvl.item() / bsize
    
    if schedname == 'plateau':
        scheduler.step(val_loss)

    with open(lossfilename, 'a') as f:
        f.write(f'{epoch+1}, {val_loss}\n')

    print(f'Epoch {epoch+1}, Validation Loss: {val_loss/valy.shape[0]}')
    print(f'eegrec: {eegloss/len(valy)}, artefactrec: {artefactloss/len(valy)}, wvl: {wvlloss/len(valy)}')
    # print(f'weights - {torch.exp(-2 * model.log_vars)}')
    valLosses.append(val_loss)
    valRecLosses.append(eegloss)

    torch.save(model, f"{filename}_epoch_{preload_epoch+epoch}.pt")

plt.figure(dpi=300)
plt.plot(np.array(trainLosses)/(trainx.shape[0]), label='Training Loss')
plt.plot(np.array(valLosses)/(valx.shape[0]), label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'{filename}_loss.png')

plt.figure(dpi=300)
plt.plot(np.array(valRecLosses)/(valy.shape[0]), label='Validation eeg Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'{filename}_eeg_val_loss.png')


print(f'minimum loss at {np.argmin(valLosses)}')