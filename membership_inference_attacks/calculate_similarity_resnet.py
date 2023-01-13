import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
# import numpy as np

import torch.nn.functional as F


class ResnetSimilarity():
    def __init__(self):
        self.train_on_gpu = torch.cuda.is_available()
        if self.train_on_gpu:
            self.model = models.resnet50(pretrained=True).cuda()
        else:
            self.model = models.resnet50(pretrained=True)

        self.model = nn.Sequential(*list(self.model.children())[:-1])
        for param in self.model.parameters():
            param.requires_grad = False
        self.transform = transforms.Compose([transforms.Resize(224),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.cos = nn.CosineSimilarity(dim=0, eps=1e-3)

    # print(self.model)

    def getMapping(self, image):
        if self.train_on_gpu:
            img = Variable(self.transform(image)).cuda()
        else:
            img = Variable(self.transform(image))
        img = img.view(1, 3, 224, 224)

        img1_feature = self.model(img)
        img1_feature = img1_feature.cpu()
        return img1_feature

    def resnetSimmilarity_C(self, image1, image2):

        img1_feature = self.getMapping(image1)
        img1_feature = img1_feature.view(-1)

        img2_feature = self.getMapping(image2)
        img2_feature = img2_feature.view(-1)

        cos_sim = self.cos(img1_feature, img2_feature)

        return torch.sum(cos_sim)

    def resnetSimmilarity_P(self, image1, image2):
        img1_feature = self.getMapping(image1)
        img1_feature = img1_feature.view(-1)

        img2_feature = self.getMapping(image2)
        img2_feature = img2_feature.view(-1)

        pairwise_dist = F.pairwise_distance(img1_feature, img2_feature)

        return torch.sum(pairwise_dist)

# pic_one = Image.open("./data/flower_data/train/1/image_06734.jpg")
# pic_two = Image.open("./data/flower_data/train/1/image_06734.jpg")
# pic_3 = Image.open("./data/download.jpeg")	

# res = ResnetSimmilarity()
# value = res.resnetSimmilarity_C(pic_one,pic_two)
# valuec = res.resnetSimmilarity_C(pic_one,pic_3)
# print(value,valuec)


# value = res.resnetSimmilarity_C(pic_one,pic_two)
# valuec = res.resnetSimmilarity_C(pic_one,pic_3)
# print(value,valuec)
