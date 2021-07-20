import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import os
model = models.mobilenet_v3_large(pretrained=True)

import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import json
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
#import pdb;pdb.set_trace()
writer = SummaryWriter('runs/0719_mobilenet_tensorboard')

def adjust_learning_rate(optimizer,lr, iter):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr_adj = lr * iter/50
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_adj

class flag_dataset(Dataset):
    def __init__(self, data_root,annotations_file, transform=None):
        super(flag_dataset,self).__init__()
        self.data_root = data_root
        #self.img_labels = pd.read_csv(annotations_file)
        self.train_file = []
        if isinstance(annotations_file,list):
            for anno in annotations_file:
                with open(anno) as f:
                    self.train_file.extend(json.load(f))
        else:
            with open(annotations_file) as f:
                self.train_file = json.load(f)
        
        self.transform = transform
        
    def __len__(self):
        return len(self.train_file)

    def __getitem__(self, idx):
        #import pdb;pdb.set_trace()
        rela_path,label,bbox = self.train_file[idx]
        img_path = os.path.join(self.data_root,rela_path)
        image = Image.open(img_path)
        #(left, upper, right, lower) = (20, 20, 100, 100)
        #im_crop = im.crop((left, upper, right, lower))
        try:
            if bbox:
                image_crop = image.crop(bbox)
            else:
                image_crop = image
            if self.transform:
                #print('do transformation...')
                image_crop = self.transform(image_crop)
        except:
            print(img_path)
        return image_crop, int(label)

train_transform=transforms.Compose([
            transforms.Resize((112,112)),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
                            ])

test_transform=transforms.Compose([
            transforms.Resize((112,112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
                            ])

batch_size = 128

data_root = '/data/flag/Images/images'
train_ann_file = '/data/flag/flag_classification_train_12cls.json'   #flag_classify_train.json'
#val_ann_file = '/data/flag/flag_classify_val.json'
test_ann_file = '/data/flag/flag_classification_test_12cls.json'

trainset = flag_dataset(data_root,train_ann_file,train_transform)
testset = flag_dataset(data_root,test_ann_file,test_transform)
trainloader = DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=4)

testloader = DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

valid_data_size = len(testset)

classes = ('china','us','uk','russia','japan','france','german','italy','australia','korea','other','logo','background')


#fc_inputs = model.fc.in_features
#model.fc = nn.Linear(fc_inputs, len(classes))
fc_inputs = model.classifier[3].in_features
model.classifier[3] = nn.Linear(fc_inputs, len(classes))

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[20,40],gamma = 0.1)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to('cuda:0')
history = []
best_acc = 0.0
best_epoch = 0
lr = 0.01
#import pdb;pdb.set_trace()
#from tqdm import tqdm
for epoch in range(1,51):  # loop over the dataset multiple times

    running_loss = 0.0
    valid_loss = 0.0
    valid_acc = 0.0
    model.train()
    
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        if epoch == 1 and i <= 50:
            adjust_learning_rate(optimizer,lr,i)

        #import pdb;pdb.set_trace()

        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f lr:%f' %
                  (epoch, i + 1, running_loss / 10,optimizer.param_groups[0]['lr']))

            writer.add_scalar('training loss',
                            running_loss / 10,
                            epoch * len(trainloader) + i)

            running_loss = 0.0
    
    scheduler.step()
        
    with torch.no_grad():
        model.eval()

        for j, (inputs, labels) in enumerate(testloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            valid_loss += loss.item() * inputs.size(0)

            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            valid_acc += acc.item() * inputs.size(0)


    avg_valid_loss = valid_loss/valid_data_size
    avg_valid_acc = valid_acc/valid_data_size

    checkpoint = {
        'epoch':epoch,
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict()
    }
    torch.save(checkpoint, os.path.join('/data/Detection_proj/resnet18_flag/weights/0719_mobilenet','{}.pt'.format(epoch)))

    if best_acc < avg_valid_acc:
        best_acc = avg_valid_acc
        best_epoch = epoch
        checkpoint = {
            'epoch':epoch,
			'model_state_dict':model.state_dict(),
			'optimizer_state_dict':optimizer.state_dict()
        }
        torch.save(checkpoint, os.path.join('/data/Detection_proj/resnet18_flag/weights/0719_mobilenet','best.pt'))


    print("Epoch: {:03d}  Validation: Loss: {:.4f}, Accuracy: {:.4f}".format(
        epoch, avg_valid_loss, avg_valid_acc*100
    ))

print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))
print('Finished Training')