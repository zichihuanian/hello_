import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict


def conv1x1(in_planes, out_planes):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=False)
class WordAttention(nn.Module):
    def __init__(self, idf, cdf):
        super(WordAttention, self).__init__()
        self.conv_context = conv1x1(cdf, idf)
        self.sm = nn.Softmax(dim=1)
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask  # batch x sourceL

    def forward(self, input, context):
        """
            input: batch x idf x ih x iw (queryL=ihxiw)
            context: batch x cdf x sourceL
        """
        ih, iw = input.size(2), input.size(3)
        queryL = ih * iw
        batch_size, sourceL = context.size(0), context.size(2)

        # --> batch x queryL x idf
        target = input.view(batch_size, -1, queryL)
        targetT = torch.transpose(target, 1, 2).contiguous()
        # batch x cdf x sourceL --> batch x cdf x sourceL x 1
        sourceT = context.unsqueeze(3)
        # --> batch x idf x sourceL
        sourceT = self.conv_context(sourceT).squeeze(3)

        # Get attention
        # (batch x queryL x idf)(batch x idf x sourceL)
        # -->batch x queryL x sourceL
        attn = torch.bmm(targetT, sourceT)
        # --> batch*queryL x sourceL
        attn = attn.view(batch_size*queryL, sourceL)
        if self.mask is not None:
            # batch_size x sourceL --> batch_size*queryL x sourceL
            mask = self.mask.repeat(queryL, 1)
            attn.data.masked_fill_(mask.data, -float('inf'))
            # print("using mask")
        attn = self.sm(attn)  # Eq. (2)
        # --> batch x queryL x sourceL
        attn = attn.view(batch_size, queryL, sourceL)
        # --> batch x sourceL x queryL
        attn = torch.transpose(attn, 1, 2).contiguous()

        # (batch x idf x sourceL)(batch x sourceL x queryL)
        # --> batch x idf x queryL
        weightedContext = torch.bmm(sourceT, attn)
        weightedContext = weightedContext.view(batch_size, -1, ih, iw)
        attn = attn.view(batch_size, -1, ih, iw)

        return weightedContext, attn


class NetG(nn.Module):
    def __init__(self, ngf=64, nz=100):
        super(NetG, self).__init__()
        self.ngf = ngf
        #self.rnn = rnn
        # self.lstm = lstm
        # layer1输入的是一个100x1x1的随机噪声, 输出尺寸(ngf*8)x4x4
        self.fc = nn.Linear(nz, ngf*8*4*4)
        self.block0 = G_Block(ngf * 8, ngf * 8)#4x4
        self.block1 = G_Block(ngf * 8, ngf * 8)#4x4
        self.block2 = G_Block(ngf * 8, ngf * 8)#8x8
        self.block2_w = G_Block_word(ngf * 8, ngf * 8)  # 32x32
        self.conv_2_1 = nn.Conv2d(ngf * 8 * 2, ngf * 8, 3, 1, 1)
        self.conv_2_2 = nn.Conv2d(ngf * 8, ngf * 8, 3, 1, 1)

        self.block3 = G_Block(ngf * 8, ngf * 8)#16x16
        self.block3_w = G_Block_word(ngf * 8, ngf * 8)  # 32x32
        self.conv_3_1 = nn.Conv2d(ngf * 8 * 2, ngf * 8, 3, 1, 1)
        self.conv_3_2 = nn.Conv2d(ngf * 8, ngf * 8, 3, 1, 1)

        self.block4 = G_Block(ngf * 8, ngf * 4)  # 32x32

        self.block5 = G_Block(ngf * 4, ngf * 2)#64x64
        # self.block5_w = G_Block_word(ngf * 4, ngf * 2)  # 64x64
        # self.conv_5_1 = nn.Conv2d(ngf * 2 * 2, ngf * 2, 3, 1, 1)
        # self.conv_5_2 = nn.Conv2d(ngf * 2, ngf * 2, 3, 1, 1)
        self.block6 = G_Block(ngf * 2, ngf * 1)#128x128
        # self.block6_w = G_Block_word(ngf * 2, ngf * 1)  # 128x128
        # self.conv_6_1 = nn.Conv2d(ngf * 1 * 2, ngf * 1, 3, 1, 1)
        # self.conv_6_2 = nn.Conv2d(ngf * 1, ngf * 1, 3, 1, 1)

        self.conv_img = nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(ngf, 3, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, x, c, word, mask):
        # x :  [bs, 100]
        # fc 输入的是一个100x1x1的随机噪声, 输出尺寸(ngf=64*8)x4x4
        # bs 8192
        out = self.fc(x)
        out = out.view(x.size(0), 8*self.ngf, 4, 4)   # bs, 512, 4, 4
        out = self.block0(out,c)   # bs, 512, 4, 4

        out = F.interpolate(out, scale_factor=2)   # bs, 512, 8, 8
        out = self.block1(out,c)   # bs, 512, 8, 8

        out = F.interpolate(out, scale_factor=2)  # bs, 512, 16, 16
        out_s = self.block2(out,c)   # bs, 512, 16, 16
        out_w = self.block2_w(out, word, mask)
        out_c = torch.cat((out_s, out_w,), dim=1)
        out = self.conv_2_1(out_c)
        out = self.conv_2_2(out)

        out = F.interpolate(out, scale_factor=2) # bs, 512, 32, 32
        out_s = self.block3(out,c)   # bs, 512, 32, 32
        out_w = self.block3_w(out, word, mask)
        out_c = torch.cat((out_s, out_w,), dim=1)
        out = self.conv_3_1(out_c)
        out = self.conv_3_2(out)


        # 从block4开始引入affine_word
        out = F.interpolate(out, scale_factor=2)  # bs, 512, 64, 64
        out = self.block4(out,c)    # bs, 256, 64, 64


        out = F.interpolate(out, scale_factor=2)    # bs, 256, 128, 128
        out = self.block5(out, c)
        # out_w = self.block5_w(out, word, mask)    # bs, 128, 128, 128
        # out_c = torch.cat((out_s, out_w,), dim=1)
        # out = self.conv_5_1(out_c)
        # out = self.conv_5_2(out)

        out = F.interpolate(out, scale_factor=2)  # bs, 128, 256, 256
        out = self.block6(out, c)    # bs, 64, 256, 256
        # out_w = self.block6_w(out, word, mask)
        # out_c = torch.cat((out_s, out_w,), dim=1)
        # out = self.conv_6_1(out_c)
        # out = self.conv_6_2(out)

        out = self.conv_img(out)   # bs, 3, 256, 256

        return out



class G_Block(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(G_Block, self).__init__()

        self.learnable_sc = in_ch != out_ch 
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.affine0 = affine(in_ch)
        self.affine1 = affine(in_ch)
        # self.affine4 = affine(in_ch)
        self.affine2 = affine(out_ch)
        self.affine3 = affine(out_ch)
        # self.affine5 = affine(out_ch)
        # self.fea_l = nn.Linear(in_ch,256)
        # self.fea_ll = nn.Linear(out_ch,256)
        self.gamma = nn.Parameter(torch.zeros(1))
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch,out_ch, 1, stride=1, padding=0)
    # noise is x, text feature is y/c
    def forward(self, x, y=None):
        return self.shortcut(x) + self.gamma * self.residual(x, y)

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        return x

    def residual(self, x, y=None):
        # bs, 512, 4, 4

        h = self.affine0(x, y)  # bs, 512, 4, 4
        h = nn.LeakyReLU(0.2,inplace=True)(h)
        


        h = self.affine1(h, y)  # bs, 512, 4, 4
        h = nn.LeakyReLU(0.2,inplace=True)(h)

        
        
        h = self.c1(h)   # bs, 512, 4, 4   这个在in != out 时候变换维度
        

        
        h = self.affine2(h, y)  # bs, 512, 4, 4
        h = nn.LeakyReLU(0.2,inplace=True)(h)

        h = self.affine3(h, y)  # bs, 512, 4, 4  全程不变维度
        h = nn.LeakyReLU(0.2,inplace=True)(h)
          
        return self.c2(h)


class G_Block_word(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(G_Block_word, self).__init__()

        self.learnable_sc = in_ch != out_ch
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.affine0 = affine_word(in_ch)
        # self.affine1 = affine_word(in_ch)
        # self.affine4 = affine_word(in_ch)
        self.affine2 = affine_word(out_ch)
        # self.affine3 = affine_word(out_ch)
        # self.affine5 = affine_word(out_ch)

        self.gamma = nn.Parameter(torch.zeros(1))
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)

    # noise is x, text feature is y/c
    def forward(self, x, y=None, mask=None):
        return self.shortcut(x) + self.gamma * self.residual(x, y, mask)

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        return x

    def residual(self, x, y=None, mask = None):


        h = self.affine0(x, y, mask)
        h = nn.LeakyReLU(0.2, inplace=True)(h)



        # h = self.affine1(h, y)
        # h = nn.LeakyReLU(0.2, inplace=True)(h)

        h = self.c1(h)


        h = self.affine2(h, y, mask)
        h = nn.LeakyReLU(0.2, inplace=True)(h)



        # h = self.affine3(h, y)
        # h = nn.LeakyReLU(0.2, inplace=True)(h)

        return self.c2(h)


class affine(nn.Module):

    def __init__(self, num_features):
        super(affine, self).__init__()

        self.fc_gamma = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(256, 256)),
            ('relu1',nn.ReLU(inplace=True)),
            ('linear2',nn.Linear(256, num_features)),
            ]))
        self.fc_beta = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(256, 256)),
            ('relu1',nn.ReLU(inplace=True)),
            ('linear2',nn.Linear(256, num_features)),
            ]))
        self._initialize()

    def _initialize(self):
        nn.init.zeros_(self.fc_gamma.linear2.weight.data)
        nn.init.ones_(self.fc_gamma.linear2.bias.data)
        nn.init.zeros_(self.fc_beta.linear2.weight.data)
        nn.init.zeros_(self.fc_beta.linear2.bias.data)

    def forward(self, x, y=None):
   
        weight = self.fc_gamma(y)
        bias = self.fc_beta(y)        

        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)

        size = x.size()
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        return weight * x + bias


def conv3x3(in_planes, out_planes, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=bias)




class affine_word(nn.Module):
    def __init__(self, channel_num):
        super(affine_word, self).__init__()

        self.w_att = WordAttention(channel_num, 256)
        self.conv = conv3x3(channel_num, 128)
        self.conv_weight = conv3x3(128, channel_num)  # weight
        self.conv_bias = conv3x3(128, channel_num)  # bias

    def forward(self, x, y=None, mask = None):
        # x img feature, y txt feature
        self.w_att.applyMask(mask)
        c_code, _ = self.w_att(x, y)
        out_code = self.conv(c_code)
        out_code_weight = self.conv_weight(out_code)
        out_code_bias = self.conv_bias(out_code)
        return x * out_code_weight + out_code_bias


class D_GET_LOGITS_att(nn.Module):
    def __init__(self, ndf):
        super(D_GET_LOGITS_att, self).__init__()
        self.df_dim = ndf

        self.joint_conv = nn.Sequential(
            nn.Conv2d(ndf * 16+256, ndf * 2, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(ndf * 2, 1, 4, 1, 0, bias=False),
        )
        self.block = resD(ndf * 16+256, ndf * 16)#4

        self.joint_conv_att = nn.Sequential(
            nn.Conv2d(ndf * 16+256, ndf * 2, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(ndf * 2, 1, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )
        self.softmax= nn.Softmax(2)
    def forward(self, out, y_):
        # y [bs, 256] out [bs, 1024, 8, 8]
        y = y_.view(-1, 256, 1, 1)  # y [bs, 256, 1, 1]
        y = y.repeat(1, 1, 8, 8)  # [bs, 256, 8, 8]
        h_c_code = torch.cat((out, y), 1)  # [bs, 1280, 8, 8]
        p = self.joint_conv_att(h_c_code)     # [bs, 1, 8, 8]
        p = self.softmax(p.view(-1,1,64))  #[bs, 1, 64]
        p = p.reshape(-1,1,8,8)   # [bs, 1, 8, 8]
        self.p = p
        p = p.repeat(1, 256, 1, 1)  # [bs, 256, 8, 8]
        y = torch.mul(y,p)
        h_c_code = torch.cat((out, y), 1)  # [bs, 1280, 8, 8]
        h_c_code = self.block(h_c_code)   # [bs, 1024, 4, 4]

        y = y_.view(-1, 256, 1, 1) # y [bs, 256, 1, 1]
        y = y.repeat(1, 1, 4, 4)        # y [bs, 256, 4, 4]
        h_c_code = torch.cat((h_c_code, y), 1)  # [bs, 1280, 4, 4]
        out = self.joint_conv(h_c_code)   # [bs, 1, 1, 1]
        return out


class D_GET_LOGITS_att_Word(nn.Module):
    def __init__(self, ndf):
        super(D_GET_LOGITS_att_Word, self).__init__()
        self.df_dim = ndf
        self.w_att = WordAttention(ndf * 16, 256)

        self.joint_conv = nn.Sequential(
            nn.Conv2d(ndf * 16, ndf * 2, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(ndf * 2, 1, 4, 1, 0, bias=False),
        )
        self.block = resD(ndf * 16+512, ndf * 16)#4

    def forward(self, out, y_):
        # out []
        c_code, _ = self.w_att(out, y_)
        h_c_code = torch.cat((out, c_code), 1)
        h_c_code = self.block(h_c_code)
        out = self.joint_conv(h_c_code)

        return out



# 定义鉴别器网络D
class NetD(nn.Module):
    def __init__(self, ndf):
        super(NetD, self).__init__()

        self.conv_img = nn.Conv2d(3, ndf, 3, 1, 1)#128
        self.block0 = resD(ndf * 1, ndf * 2)#64
        self.block1 = resD(ndf * 2, ndf * 4)#32
        self.block2 = resD(ndf * 4, ndf * 8)#16
        self.block3 = resD(ndf * 8, ndf * 16)#8
        self.block4 = resD(ndf * 16, ndf * 16)#4
        # self.block5 = resD(ndf * 16, ndf * 16)#4

        self.COND_DNET = D_GET_LOGITS_att(ndf)


    def forward(self,x):

        out = self.conv_img(x)
        out = self.block0(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        # out = self.block5(out)

        return out

# 定义鉴别器网络D
class NetD_word(nn.Module):
    def __init__(self, ndf):
        super(NetD_word, self).__init__()

        self.conv_img = nn.Conv2d(3, ndf, 3, 1, 1)#128
        self.block0 = resD(ndf * 1, ndf * 2)#64
        self.block1 = resD(ndf * 2, ndf * 4)#32
        self.block2 = resD(ndf * 4, ndf * 8)#16
        self.block3 = resD(ndf * 8, ndf * 16)#8
        self.block4 = resD(ndf * 16, ndf * 16)#4
        # self.block5 = resD(ndf * 16, ndf * 16)#4

        self.COND_DNET_word = D_GET_LOGITS_att_Word(ndf)

    def forward(self,x):

        out = self.conv_img(x)
        out = self.block0(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        # out = self.block5(out)

        return out



class resD(nn.Module):
    def __init__(self, fin, fout, downsample=True):
        super().__init__()
        self.downsample = downsample
        self.learned_shortcut = (fin != fout)
        self.conv_r = nn.Sequential(
            nn.Conv2d(fin, fout, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(fout, fout, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_s = nn.Conv2d(fin,fout, 1, stride=1, padding=0)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, c=None):
        return self.shortcut(x)+self.gamma*self.residual(x)

    def shortcut(self, x):
        if self.learned_shortcut:
            x = self.conv_s(x)
        if self.downsample:
            return F.avg_pool2d(x, 2)
        return x

    def residual(self, x):
        return self.conv_r(x)












