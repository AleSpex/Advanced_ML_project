import sys
sys.path.append('../loader')
from datasets_ import Dataset
import torchvision.transforms as transforms
import torch


def _dataset_info(args,txt_labels):

    with open(txt_labels, 'r') as f:
        images_list = f.readlines()

    file_names = []
    labels = []
    for row in images_list:
        row = row.split(' ')
        file_names.append(args.path_to_dataset+row[0])
        labels.append(int(row[1]))

    return file_names, labels

def get_test_transformers():
    img_tr = [transforms.Resize(256),
              transforms.CenterCrop(224),
              transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    return transforms.Compose(img_tr)



def dataset_read_eval(target,args):
    img_transformer = get_test_transformers()
    name_train,labels_train= _dataset_info(args,target)
    dataset = Dataset(name_train, labels_train,img_transformer=img_transformer)
    target_test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    return target_test_loader




