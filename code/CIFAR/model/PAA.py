import kornia
from kornia.augmentation import RandomErasing, RandomSharpness
from kornia.augmentation import RandomPosterize, RandomEqualize, RandomAffine
from kornia.augmentation import RandomRotation,RandomGrayscale, ColorJitter
from kornia.augmentation import Normalize
import numpy as np
import random
import torch
from torch.distributions import Categorical

from torchvision import transforms
import time

# Augmentation action

normalize_mean = torch.tensor((125.3, 123.0, 113.9))/255.0
normalize_std = torch.tensor((63.0, 62.1, 66.7))/255.0

# randomerasing = RandomErasing(p=1., scale=(0.09, 0.36), ratio=(0.5, 1/0.5), same_on_batch=False)
# sharpness = RandomSharpness(sharpness=0.5, same_on_batch=False)
# # randomerasing = RandomErasing(p=1., scale=(.2, .4), ratio=(.1, 1/.3), same_on_batch=False)
# # sharpness = RandomSharpness(sharpness=1., same_on_batch=False)
# posterize = RandomPosterize(bits=3, same_on_batch=False)
# equalize = RandomEqualize()
# shear = RandomAffine(degrees=0., shear=(10, 20))
# translate = RandomAffine(degrees=0.,translate=(0.3, 0.4), same_on_batch = False)
# brightness = ColorJitter(brightness=(0.5, 0.9))
# contrast = ColorJitter(contrast=(0.5, 0.9))
# color = ColorJitter(hue=(-0.3, 0.3)) # time consuming but feel useful
# rotation= RandomRotation(degrees=60.0)
# gray = RandomGrayscale(p=1.0)

randomerasing = RandomErasing(p=1., scale=(0.09, 0.36), ratio=(0.5, 1/0.5), same_on_batch=False)
# randomerasing = RandomErasing(p=1., scale=(.2, .4), ratio=(.1, 1/.3), same_on_batch=False)
posterize = RandomPosterize(bits=3, same_on_batch=False)
shear = RandomAffine(degrees=0., shear=(30, 30), same_on_batch=False)
translate = RandomAffine(degrees=0., translate=(0.4, 0.4), same_on_batch=False)
brightness = ColorJitter(brightness=(0.5, 0.95), same_on_batch=False)
contrast = ColorJitter(contrast=(0.5, 0.95), same_on_batch=False)
rotation= RandomRotation(degrees=30.0, same_on_batch=False)
gray = RandomGrayscale(p=1.0, same_on_batch=False)
color = ColorJitter(hue=(-0.3, 0.3)) # time consuming but feel useful
equalize = RandomEqualize(p=1.0, same_on_batch=False)
sharpness = RandomSharpness(sharpness=.5, same_on_batch=False)


def invert(imagetensor):
    imagetensor = 1-imagetensor
    return imagetensor
def dropout(imagetensor):
    mean=imagetensor.mean()
    imagetensor[:,:,:]=mean
    return imagetensor
def mixup(imagetensor):
    lower_bound = 0.5
    lam = (1-lower_bound) * random.random() + lower_bound   #! ADD by ry
    for i in range(imagetensor.size(0)-1):
        imagetensor[i]=lam*imagetensor[i]+(1-lam)*imagetensor[i+1]
    imagetensor[-1]=lam*imagetensor[-1]+(1-lam)*imagetensor[0]
    return imagetensor
def cutmix(imagetensor):
    for i in range(imagetensor.size(0) - 1):
        imagetensor[i] = imagetensor[i + 1]
    imagetensor[-1] = imagetensor[0]
    return imagetensor
def cropimage(image, i, j, k, patch_size=16):
    region = image[i,:, j * patch_size:j * patch_size + (patch_size-1), k * patch_size:k * patch_size + patch_size-1]
    return region
def cutout_tensor(image, length=16):
    mean = 0
    top = np.random.randint(0 - length // 2, 32 - length)
    left = np.random.randint(0 - length // 2, 32 - length)
    bottom = top + length
    right = left + length
    top = 0 if top < 0 else top
    left = 0 if left < 0 else left
    image[:, :, top:bottom, left:right] = mean
    return image

ops = [
        randomerasing, 
        posterize, 
        dropout, 
        cutmix, 
        shear, 
        mixup, 
        translate,
        brightness, 
        contrast, 
        rotation, 
        gray,
        invert,
        color,  
        equalize,   
        sharpness,    #! to be done
        ]
num_ops = len(ops)

def patch_auto_augment(image, operations, batch_size, size=2, patch_size=16):
    tensorlist = [ [] for _ in range(num_ops)] 
    location = [ [] for _ in range(num_ops)]
    sum_num = []

    for i in range(batch_size): 
        for j in range(size):
            for k in range(size):
                '''First,randomly select probability.
                   Then if act , RL select operation
                '''
                # probability = 1.0
                probability = random.random()
                operation = operations[i,j,k].item() #0-13 include 13 ,14 in total
                if random.random()<probability:
                    tensorlist[operation].append(cropimage(image,i,j,k).squeeze())
                    location[operation].append((i,j,k))

    # output = torch.ones_like(image)
    output = image.clone().detach()
    for i in range(num_ops): # 13 without equalize, 12 without sharpness 
        if len(tensorlist[i])==0 :
            continue
        # aftertensor = ops[i](torch.stack(tensorlist[i]))  # time: 0.2 - 0.3
        try:
            if ops[i] in [sharpness]:
                aftertensor = ops[i](torch.stack(tensorlist[i]).cpu()).cuda()
            else:
                aftertensor = ops[i](torch.stack(tensorlist[i]).cuda()) # time: 0.2 - 0.3
        except:
            aftertensor = torch.stack(tensorlist[i]).cuda()
            print('an error accured in ops[i]: {}'.format(ops[i]))
            pass

        for j in range(aftertensor.size(0)):
            t= location[i][j]
            # image[t[0], :, t[1]* patch_size:t[1] * patch_size + (patch_size-1), 
            #     t[2]* patch_size:t[2]* patch_size + patch_size - 1] = aftertensor[j] # 0.01
            output[t[0], :, t[1]* patch_size:t[1] * patch_size + (patch_size-1), 
                t[2]* patch_size:t[2]* patch_size + patch_size - 1] = aftertensor[j] # 0.01

    output = Normalize(normalize_mean, normalize_std)(output)
    # output = RandomErasing(p=1.0, scale=(0.09, 0.36), ratio=(0.5, 1/0.5), same_on_batch=False)(output)
    # output = RandomErasing(p=1., scale=(.2, .4), ratio=(.1, 1/.3), same_on_batch=False)(output)
    # output = cutout_tensor(output)    
    return output
    # return image

# choose action
def rl_choose_action(images, a2cmodel):
    dist, state_value = a2cmodel(images) # dist N 14 4 4
    dist = dist.permute(0,2,3,1).contiguous()# N 4 4 14
    operation_dist = Categorical(dist)
    operations = operation_dist.sample() # N 4 4
    operation_logprob = operation_dist.log_prob(operations).mean()
    operation_entropy = operation_dist.entropy().mean()
    return operations, operation_logprob, operation_entropy, state_value

# loss
def a2closs(operation_logprob , operation_entropy, state_value, training_loss):
    advantage = (training_loss - state_value).mean()
    actor_loss = -(operation_logprob * advantage)
    critic_loss = advantage.pow(2)
    a2closs = actor_loss + 0.5 * critic_loss - 0.001 * operation_entropy
    # ? hyper parameter need adjust
    return a2closs

# visualize
def saveimage(tensorlist, str):
    savepath = './visualize'
    import os
    if not os.path.exists(savepath):
        os.makedirs(savepath, exist_ok=True)
    to_pil_image = transforms.ToPILImage()
    for i in range(tensorlist.size(0)): #100 tensorlist.size(0)
        image=tensorlist[i].squeeze(0).cpu()
        imageresult=to_pil_image(image)
        imageresult.save(savepath + "/%s%d.jpg"%(str,i))

# time_recoder
class TimeRecoder():
    '''
    Usage:
        time_recoder = TimeRecoder()
        for ...
            time_recoder.insert()
            ...code...
            time_recoder.insert()
            ...code...
            time_recoder.insert()
            time_recoder.fresh()
        time_recoder.display()
    '''
    def __init__(self):
        self.last_time = 0
        self.time_2d_list = [[]]

    def insert(self):
        if self.last_time == 0:
            self.last_time = time.time()
        else:
            time_gap = time.time() - self.last_time
            self.last_time = time.time()
            self.time_2d_list[-1].append(time_gap)

    def fresh(self):
        self.last_time = 0
        self.time_2d_list.append([])

    def display(self, is_sum=False):
        self.time_2d_list.pop()
        self.time_2d_list_T = map(list, zip(*self.time_2d_list))
        if is_sum:
            results = [sum(each) for each in self.time_2d_list_T]
        else:
            results = [sum(each)/len(each) for each in self.time_2d_list_T]
        for i in range(len(results)):
            print('time cost in step {} to {} is {}'.format(i, i+1, results[i]))
