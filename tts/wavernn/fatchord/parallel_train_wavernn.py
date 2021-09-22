import time
import random
import logging
import os

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch import optim
import torch.nn.functional as F
from utils.display import stream, simple_table
from utils.dataset import get_vocoder_datasets_ddp
from utils.distribution import discretized_mix_logistic_loss
from utils import hparams as hp
from models.fatchord_version import WaveRNN
from gen_wavernn import gen_testset
from utils.paths import Paths
import argparse
from utils import data_parallel_workaround
from utils.checkpoints import save_checkpoint, restore_checkpoint


logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(os.path.basename(__file__))


def train(rank, world_size, args):
    hp.configure(args.hp_file)  # load hparams from file
    if args.lr is None:
        args.lr = hp.voc_lr
    if args.batch_size is None:
        args.batch_size = hp.voc_batch_size

    paths = Paths(hp.data_path, hp.voc_model_id, hp.tts_model_id)

    batch_size = args.batch_size
    force_train = args.force_train
    train_gta = args.gta
    lr = args.lr

    torch.manual_seed(0)

    torch.cuda.set_device(rank)
    device = f"cuda:{rank}

    print('\nInitialising Model...\n')

    # Instantiate WaveRNN Model
    voc_model = WaveRNN(rnn_dims=hp.voc_rnn_dims,
                        fc_dims=hp.voc_fc_dims,
                        bits=hp.bits,
                        pad=hp.voc_pad,
                        upsample_factors=hp.voc_upsample_factors,
                        feat_dims=hp.num_mels,
                        compute_dims=hp.voc_compute_dims,
                        res_out_dims=hp.voc_res_out_dims,
                        res_blocks=hp.voc_res_blocks,
                        hop_length=hp.hop_length,
                        sample_rate=hp.sample_rate,
                        mode=hp.voc_mode).to(device)
    voc_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(voc_model)
    voc_model = torch.nn.parallel.DistributedDataParallel(voc_model, device_ids=[rank])

    # Check to make sure the hop length is correctly factorised
    assert np.cumprod(hp.voc_upsample_factors)[-1] == hp.hop_length

    optimizer = optim.Adam(voc_model.parameters())
    restore_checkpoint('voc', paths, voc_model, optimizer, create_if_missing=True)

    train_set, test_set = get_vocoder_datasets_ddp(paths.data, batch_size, train_gta, rank, world_size)

    total_steps = 10_000_000 if force_train else hp.voc_total_steps

    simple_table([('Remaining', str((total_steps - voc_model.get_step())//1000) + 'k Steps'),
                  ('Batch Size', batch_size),
                  ('LR', lr),
                  ('Sequence Len', hp.voc_seq_len),
                  ('GTA Train', train_gta)])

    loss_func = F.cross_entropy if voc_model.mode == 'RAW' else discretized_mix_logistic_loss

    dist.barrier()

    voc_train_loop(paths, voc_model, loss_func, optimizer, train_set, test_set, lr, total_steps, rank, world_size)

def main():
    # Parse Arguments
    parser = argparse.ArgumentParser(description='Train WaveRNN Vocoder')
    parser.add_argument('--lr', '-l', type=float,  help='[float] override hparams.py learning rate')
    parser.add_argument('--batch_size', '-b', type=int, help='[int] override hparams.py batch size')
    parser.add_argument('--force_train', '-f', action='store_true', help='Forces the model to train past total steps')
    parser.add_argument('--gta', '-g', action='store_true', help='train wavernn on GTA features')
    parser.add_argument('--force_cpu', '-c', action='store_true', help='Forces CPU-only training, even when in CUDA capable environment')
    parser.add_argument('--hp_file', metavar='FILE', default='hparams.py', help='The file to use for the hyperparameters')
    args = parser.parse_args()



    if not args.force_cpu and torch.cuda.is_available():
        device = torch.device('cuda')
        if args.batch_size % torch.cuda.device_count() != 0:
            