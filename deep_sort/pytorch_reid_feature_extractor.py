import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2
#import logging


#from fastreid.config import get_cfg
#from fastreid.engine import DefaultTrainer
#from fastreid.utils.checkpoint import Checkpointer



class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out,is_downsample=False):
        super(BasicBlock,self).__init__()
        self.is_downsample = is_downsample
        if is_downsample:
            self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=2, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(c_out,c_out,3,stride=1,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        if is_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=2, bias=False),
                nn.BatchNorm2d(c_out)
            )
        elif c_in != c_out:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=1, bias=False),
                nn.BatchNorm2d(c_out)
            )
            self.is_downsample = True

    def forward(self,x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        if self.is_downsample:
            x = self.downsample(x)
        return F.relu(x.add(y),True)

def make_layers(c_in,c_out,repeat_times, is_downsample=False):
    blocks = []
    for i in range(repeat_times):
        if i ==0:
            blocks += [BasicBlock(c_in,c_out, is_downsample=is_downsample),]
        else:
            blocks += [BasicBlock(c_out,c_out),]
    return nn.Sequential(*blocks)

class Net(nn.Module):
    def __init__(self, num_classes=751 ,reid=False):
        super(Net,self).__init__()
        # 3 128 64
        self.conv = nn.Sequential(
            nn.Conv2d(3,64,3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.Conv2d(32,32,3,stride=1,padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2,padding=1),
        )
        # 32 64 32
        self.layer1 = make_layers(64,64,2,False)
        # 32 64 32
        self.layer2 = make_layers(64,128,2,True)
        # 64 32 16
        self.layer3 = make_layers(128,256,2,True)
        # 128 16 8
        self.layer4 = make_layers(256,512,2,True)
        # 256 8 4
        self.avgpool = nn.AvgPool2d((8,4),1)
        # 256 1 1 
        self.reid = reid
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        # B x 128
        if self.reid:
            x = x.div(x.norm(p=2,dim=1,keepdim=True))
            return x
        # classifier
        x = self.classifier(x)
        return x


class NetOriginal(nn.Module):
    def __init__(self, num_classes=625 ,reid=False):
        super(Net,self).__init__()
        # 3 128 64
        self.conv = nn.Sequential(
            nn.Conv2d(3,32,3,stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.Conv2d(32,32,3,stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.MaxPool2d(3,2,padding=1),
        )
        # 32 64 32
        self.layer1 = make_layers(32,32,2,False)
        # 32 64 32
        self.layer2 = make_layers(32,64,2,True)
        # 64 32 16
        self.layer3 = make_layers(64,128,2,True)
        # 128 16 8
        self.dense = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(128*16*8, 128),
            nn.BatchNorm1d(128),
            nn.ELU(inplace=True)
        )
        # 256 1 1 
        self.reid = reid
        self.batch_norm = nn.BatchNorm1d(128)
        self.classifier = nn.Sequential(
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = x.view(x.size(0),-1)
        if self.reid:
            x = self.dense[0](x)
            x = self.dense[1](x)
            x = x.div(x.norm(p=2,dim=1,keepdim=True))
            return x
        x = self.dense(x)
        # B x 128
        # classifier
        x = self.classifier(x)
        return x





class Extractor(object):
    def __init__(self, model_path, use_cuda=True):
        self.net = Net(reid=True)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)['net_dict']
        self.net.load_state_dict(state_dict)
        #logger = logging.getLogger("root.tracker")
        #logger.info("Loading weights from {}... Done!".format(model_path))
        self.net.to(self.device)
        self.size = (64, 128)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        


    def _preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32)/255., size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch


    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy()


""" class FastReIDExtractor(object):
    def __init__(self, model_config, model_path, use_cuda=True):
        cfg = get_cfg()
        cfg.merge_from_file(model_config)
        cfg.MODEL.BACKBONE.PRETRAIN = False
        self.net = DefaultTrainer.build_model(cfg)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"

        Checkpointer(self.net).load(model_path)
        logger = logging.getLogger("root.tracker")
        logger.info("Loading weights from {}... Done!".format(model_path))
        self.net.to(self.device)
        self.net.eval()
        height, width = cfg.INPUT.SIZE_TEST
        self.size = (width, height)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    
    def _preprocess(self, im_crops):
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32)/255., size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch


    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy() """




def xywh_to_xyxy(bbox_xywh, height, width):
    x,y,w,h = bbox_xywh
    x1 = max(int(x-w/2),0)
    x2 = min(int(x+w/2),width-1)
    y1 = max(int(y-h/2),0)
    y2 = min(int(y+h/2),height-1)
    return x1,y1,x2,y2

    
def get_features(extractor, bbox_xywh, ori_img):
    height, width = ori_img.shape[:2]
    im_crops = []
    for box in bbox_xywh:
        x1,y1,x2,y2 = xywh_to_xyxy(box, height, width)
        im = ori_img[y1:y2,x1:x2]
        im_crops.append(im)
    if im_crops:
        features = extractor(im_crops)
    else:
        features = np.array([])
    return features