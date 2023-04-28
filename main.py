from __future__ import print_function

from miscc.utils import mkdir_p
from miscc.config import cfg, cfg_from_file

from datasets import TextDataset
from datasets import prepare_data
#for flower dataset, please use the fllowing dataset files
#from datasets_flower import TextDataset
#from datasets_flower import prepare_data
from DAMSM import RNN_ENCODER,CustomLSTM
from attn_map import sampling_attn
import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from model import NetG,NetD,NetD_word
import torchvision.utils as vutils

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

import multiprocessing
multiprocessing.set_start_method('spawn', True)

UPDATE_INTERVAL = 200
def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/bird_e.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args



def sampling(text_encoder, netG, dataloader,device):
    
    model_dir = cfg.TRAIN.NET_G
    split_dir = 'valid'
    # Build and load the generator
    # for coco wrap netG with DataParallel because it's trained on two 3090
    #    netG = nn.DataParallel(netG).cuda()
    epoch = 450
    netG.load_state_dict(torch.load('../models/%s/netG_%s.pth'%(cfg.CONFIG_NAME, epoch)))
    
    netG.eval()

    batch_size = cfg.TRAIN.BATCH_SIZE
    s_tmp = model_dir
    save_dir = '%s/%s/%s' % (s_tmp, epoch, split_dir)
    mkdir_p(save_dir)
    cnt = 0
    with torch.no_grad():
        for i in range(11):  # (cfg.TEXT.CAPTIONS_PER_IMAGE):
            for step, data in enumerate(dataloader, 0):
                imags, captions, cap_lens, class_ids, keys = prepare_data(data)
                cnt += batch_size
                if step % 100 == 0:
                    print('step: ', step)
                # if step > 50:
                #     break
                hidden = text_encoder.init_hidden(batch_size)
                # words_embs: batch_size x nef x seq_len
                # sent_emb: batch_size x nef
                words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                mask = (captions == 0)
                num_words = words_embs.size(2)
                if mask.size(1) > num_words:
                    mask = mask[:, :num_words]
                #######################################################
                # (2) Generate fake images
                ######################################################
                with torch.no_grad():
                    noise = torch.randn(batch_size, 100)
                    noise = noise.to(device)
                    # netG.lstm.init_hidden(noise)

                    fake_imgs = netG(noise, sent_emb, words_embs, mask)
                for j in range(batch_size):
                    s_tmp = '%s/single/%s' % (save_dir, keys[j])
                    folder = s_tmp[:s_tmp.rfind('/')]
                    if not os.path.isdir(folder):
                        print('Make a new folder: ', folder)
                        mkdir_p(folder)
                    im = fake_imgs[j].data.cpu().numpy()
                    # [-1, 1] --> [0, 255]
                    im = (im + 1.0) * 127.5
                    im = im.astype(np.uint8)
                    im = np.transpose(im, (1, 2, 0))
                    im = Image.fromarray(im)
                    fullpath = '%s_%3d.png' % (s_tmp, i)
                    im.save(fullpath)

                if (cnt >= 30000):
                    break





def train(dataloader,netG,netD, netD_word, text_encoder,optimizerG,optimizerD,optimizerD_word,state_epoch,batch_size,device):
    mkdir_p('../models/%s' % (cfg.CONFIG_NAME))
  
    for epoch in range(state_epoch+1, cfg.TRAIN.MAX_EPOCH+1):
        torch.cuda.empty_cache()
        
        for step, data in enumerate(dataloader, 0):
            # torch.cuda.empty_cache()
            
            imags, captions, cap_lens, class_ids, keys = prepare_data(data)
            hidden = text_encoder.init_hidden(batch_size)
            # words_embs: batch_size x nef x seq_len
            # sent_emb: batch_size x nef
            words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
            words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
            mask = (captions == 0)
            num_words = words_embs.size(2)
            if mask.size(1) > num_words:
                mask = mask[:, :num_words]


            # ————————————————————————————————————————————————————————
            # D: sentence
            # ————————————————————————————————————————————————————————
            imgs=imags[0].to(device)
            real_features = netD(imgs)
            output = netD.module.COND_DNET(real_features, sent_emb)
            errD_real = torch.nn.ReLU()(1.0 - output).mean()

            output = netD.module.COND_DNET(real_features[:(batch_size - 1)], sent_emb[1:batch_size])
            errD_mismatch = torch.nn.ReLU()(1.0 + output).mean()

            # synthesize fake images
            noise = torch.randn(batch_size, 100)
            noise=noise.to(device)
            # netG.lstm.init_hidden(noise)
            #
            fake = netG(noise, sent_emb, words_embs, mask)
            # G does not need update with D
            fake_features = netD(fake.detach())
            errD_fake = netD.module.COND_DNET(fake_features,sent_emb)
            errD_fake = torch.nn.ReLU()(1.0 + errD_fake).mean()

            errD = errD_real + (errD_fake + errD_mismatch)/2.0
            optimizerD.zero_grad()
            optimizerG.zero_grad()
            errD.backward()
            optimizerD.step()

            # ————————————————————————————————————————————————————————
            # D: word
            # ————————————————————————————————————————————————————————
            imgs = imags[0].to(device)
            real_features = netD_word(imgs)
            output_word = netD_word.module.COND_DNET_word(real_features, words_embs)
            errD_real_word = torch.nn.ReLU()(1.0 - output_word).mean()

            output_word = netD_word.module.COND_DNET_word(real_features[:(batch_size - 1)], words_embs[1:batch_size])
            errD_mismatch_word = torch.nn.ReLU()(1.0 + output_word).mean()

            fake_features = netD_word(fake.detach())
            errD_fake_word = netD_word.module.COND_DNET_word(fake_features, words_embs)
            errD_fake_word = torch.nn.ReLU()(1.0 + errD_fake_word).mean()
            errD_word = errD_real_word + (errD_fake_word + errD_mismatch_word) / 2.0
            optimizerD_word.zero_grad()
            optimizerG.zero_grad()
            errD_word.backward()
            optimizerD_word.step()

            # ————————————————————————————————————————————————————————
            # #MA-GP: sentence 和DF-GAN的区别就是这里的out经过论文自己设计的spatial attention，然后d-loss-gp系数为2
            # ————————————————————————————————————————————————————————
            interpolated = (imgs.data).requires_grad_()
            sent_inter = (sent_emb.data).requires_grad_()

            features = netD(interpolated)
            out = netD.module.COND_DNET(features, sent_inter)
            grads = torch.autograd.grad(outputs=out,
                                    inputs=(interpolated, sent_inter),
                                    grad_outputs=torch.ones(out.size()).cuda(),
                                    retain_graph=True,
                                    create_graph=True,
                                    only_inputs=True)
            grad0 = grads[0].view(grads[0].size(0), -1)
            grad1 = grads[1].view(grads[1].size(0), -1)
            grad = torch.cat((grad0, grad1), dim=1)
            grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
            d_loss_gp = torch.mean((grad_l2norm) ** 6)
            d_loss = 2.0 * d_loss_gp
            optimizerD.zero_grad()
            optimizerG.zero_grad()
            d_loss.backward()
            optimizerD.step()

            # ————————————————————————————————————————————————————————
            # #MA-GP: word
            # ————————————————————————————————————————————————————————

            interpolated = (imgs.data).requires_grad_()
            word_inter = (words_embs.data).requires_grad_()

            features = netD_word(interpolated)
            out_word = netD_word.module.COND_DNET_word(features, word_inter)
            grads_word = torch.autograd.grad(outputs=out_word,
                                        inputs=(interpolated, word_inter),
                                        grad_outputs=torch.ones(out_word.size()).cuda(),
                                        retain_graph=True,
                                        create_graph=True,
                                        only_inputs=True)
            grad0_word = grads_word[0].view(grads_word[0].size(0), -1)
            grad1_word = grads_word[1].contiguous().view(grads_word[1].size(0), -1)
            grad_word = torch.cat((grad0_word, grad1_word), dim=1)
            grad_l2norm_word = torch.sqrt(torch.sum(grad_word ** 2, dim=1))
            d_loss_gp_word = torch.mean((grad_l2norm_word) ** 6)
            d_loss_word = 2.0 * d_loss_gp_word
            optimizerD_word.zero_grad()
            optimizerG.zero_grad()
            d_loss_word.backward()
            optimizerD_word.step()


            # update G_sentence
            features = netD(fake)
            output = netD.module.COND_DNET(features, sent_emb)
            errG = - output.mean()


            # update G_word
            features = netD_word(fake)
            output_word = netD_word.module.COND_DNET_word(features, words_embs)
            errG_word = - output_word.mean()
            err_total = errG + errG_word
            optimizerG.zero_grad()
            optimizerD.zero_grad()
            optimizerD_word.zero_grad()
            err_total.backward()
            optimizerG.step()


            if (step + 1) % 200 == 0:
                vutils.save_image(fake.data,
                                  '%s/fake_samples_epoch_%03d_step%03d.png' % (
                                  '/home/sunhaoran/A_BIYE_CODE/RAT_loss/DPF-bird/imgs', epoch, step),
                                  normalize=True)


            print('[%d/%d][%d/%d] Loss_D: %.3f Loss_G %.3f'
                % (epoch, cfg.TRAIN.MAX_EPOCH, step, len(dataloader), errD.item(), errG.item()))

        vutils.save_image(fake.data,
                        '%s/fake_samples_epoch_%03d.png' % ('/home/sunhaoran/A_BIYE_CODE/RAT_loss/DPF-bird/imgs', epoch),
                        normalize=True)

        if epoch%10==0:
            torch.save(netG.state_dict(), '/home/sunhaoran/A_BIYE_CODE/RAT_loss/DPF-bird/models/%s/netG_%03d.pth' % (cfg.CONFIG_NAME, epoch))
            torch.save(netD.state_dict(), '/home/sunhaoran/A_BIYE_CODE/RAT_loss/DPF-bird/models/%s/netD_%03d.pth' % (cfg.CONFIG_NAME, epoch))
            torch.save(netD_word.state_dict(), '/home/sunhaoran/A_BIYE_CODE/RAT_loss/DPF-bird/models/%s/netD_word_%03d.pth' % (cfg.CONFIG_NAME, epoch))

    return count




if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id == -1:
        cfg.CUDA = False
    else:
        cfg.GPU_ID = args.gpu_id

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = 100
        #args.manualSeed = random.randint(1, 10000)
    print("seed now is : ",args.manualSeed)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    ##########################################################################
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '../output/%s_%s_%s' % \
        (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    torch.cuda.set_device(cfg.GPU_ID)
    cudnn.benchmark = True

    # Get data loader ##################################################
    imsize = cfg.TREE.BASE_SIZE
    batch_size = cfg.TRAIN.BATCH_SIZE
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    if cfg.B_VALIDATION:
        dataset = TextDataset(cfg.DATA_DIR, 'test',
                                base_size=cfg.TREE.BASE_SIZE,
                                transform=image_transform)
        print(dataset.n_words, dataset.embeddings_num)
        assert dataset
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, drop_last=True,
            shuffle=True, num_workers=int(cfg.WORKERS))
    else:     
        dataset = TextDataset(cfg.DATA_DIR, 'train',
                            base_size=cfg.TREE.BASE_SIZE,
                            transform=image_transform)
        print(dataset.n_words, dataset.embeddings_num)
        assert dataset
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, drop_last=True,
            shuffle=True, num_workers=int(cfg.WORKERS))

    # # validation data #

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # lstm = CustomLSTM(256, 256)

    netG = NetG(cfg.TRAIN.NF, 100).to(device)
    netD = NetD(cfg.TRAIN.NF).to(device)
    netD_word = NetD_word(cfg.TRAIN.NF).to(device)

    text_encoder = RNN_ENCODER(dataset.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
    state_dict = torch.load(cfg.TEXT.DAMSM_NAME, map_location=lambda storage, loc: storage)
    text_encoder.load_state_dict(state_dict)
    text_encoder.cuda()

    for p in text_encoder.parameters():
        p.requires_grad = False
    text_encoder.eval()    

    state_epoch=0

    print('CUDA count', torch.cuda.device_count())
    netG = nn.DataParallel(netG, device_ids=range(torch.cuda.device_count()))
    netD = nn.DataParallel(netD, device_ids=range(torch.cuda.device_count()))
    netD_word = nn.DataParallel(netD_word, device_ids=range(torch.cuda.device_count()))
    print('using parallel training')


    optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0001, betas=(0.0, 0.9))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0004, betas=(0.0, 0.9))
    optimizerD_word = torch.optim.Adam(netD_word.parameters(), lr=0.0004, betas=(0.0, 0.9))


    if cfg.B_VALIDATION:
        # count = sampling(text_encoder, netG, dataloader,device)  # generate images for the whole valid dataset
        # print('state_epoch:  %d'%(state_epoch))
        cout = sampling(text_encoder, netG, netD, netD_word, dataloader, device)
    else:
        
        # count = train(dataloader,netG,netD, netD_word, text_encoder,optimizerG,optimizerD, optimizerD_word, state_epoch,batch_size,device)
        # count = train(dataloader,netG,netD, netD_word, text_encoder,optimizerG,optimizerD, optimizerD_word, state_epoch,batch_size,device)
        # gen_example(text_encoder, netG, dataloader,device)
        cout = sampling_attn(text_encoder, netG, netD, netD_word, dataloader, device)


        