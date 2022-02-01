from __future__ import print_function
import argparse
import torch
import sys
import torch.nn as nn
import numpy as np
sys.path.append('./model')
sys.path.append('./datasets')
sys.path.append('./metric')
from model.build_gen import  Generator,Standard_Classifier
from tqdm import tqdm
from numpy import dot
from model.triplet_match.model import TripletMatch


from datasets.dataset_read import dataset_read_eval


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

def cosine_sim(a,b):
    return ((dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))+1)


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Implementation')

parser.add_argument('--target', type=str, default='ArtPainting', metavar='N', help='target dataset')
parser.add_argument('--path_to_txt', type=str,default='/.../DomainToText_AMLProject/data/PACS', help='path to the txt files')
parser.add_argument('--path_to_dataset', type=str, default='/.../DomainToText_AMLProject/',help='path to the dataset')
parser.add_argument('--num_classes', type=int, default=7, help='size for the crop')
parser.add_argument('--gpu', type=int, default=0,help='gpu index')
args = parser.parse_args()

torch.cuda.set_device(args.gpu)


def main():

    print(args)
    target_txt = args.path_to_txt + '/' + args.target + '.txt'

    # dataloaders creation
    dataset_test_target = dataset_read_eval(target_txt, args)

    sources = ['ArtPainting','Cartoon','Photo','Sketch']
    sources.remove(args.target)
    print('Sources ' + sources[0] + ', ' + sources[1] + ', ' + sources[2])


    ########################## Source1
    # Feature Generator
    G1 = Generator().cuda()
    # Object Classifier
    C1 = Standard_Classifier(args).cuda()
    G1.load_state_dict(torch.load('outputs/SingleSource_'+sources[0] + '/G.pkl'))
    C1.load_state_dict(torch.load('outputs/SingleSource_'+sources[0] + '/C.pkl'))

    print('Model of %s loaded ' % (sources[0]))

    G1.eval()
    C1.eval()


    ########################## Source2
    # Feature Generator
    G2 = Generator().cuda()
    # Object Classifier
    C2 = Standard_Classifier(args).cuda()
    G2.load_state_dict(torch.load('outputs/SingleSource_'+sources[1] + '/G.pkl'))
    C2.load_state_dict(torch.load('outputs/SingleSource_'+sources[1] + '/C.pkl'))

    print('Model of %s loaded ' % (sources[1]))

    G2.eval()
    C2.eval()


    ########################## Source3
    # Feature Generator
    G3 = Generator().cuda()
    # Object Classifier
    C3 = Standard_Classifier(args).cuda()

    G3.load_state_dict(torch.load('outputs/SingleSource_'+sources[2] + '/G.pkl'))
    C3.load_state_dict(torch.load('outputs/SingleSource_'+sources[2] + '/C.pkl'))
    print('Model of %s loaded ' % (sources[2]))

    G3.eval()
    C3.eval()


    ########################### Computation of sources domain embeddings
    vec_dim = 256
    resnet101_texture_model = TripletMatch(vec_dim=vec_dim,distance='cos', img_feats=(2, 4))
    model_path = 'outputs/triplet_match/BEST_checkpoint.pth'

    resnet101_texture_model = resnet101_texture_model.cuda()
    resnet101_texture_model.load_state_dict(torch.load(model_path), strict=False)
    print('Model pretrained on Textures loaded')
    resnet101_texture_model.eval()


    ###### sources domain embeddings
    sources_domain_embeddings_textual = np.zeros((3, vec_dim))


    for n_source, source in enumerate(sources):

        source_txt = args.path_to_txt + '/' + source + '.txt'
        dataset_source = dataset_read_eval(source_txt, args)
        with torch.no_grad():
            for it, (img, label) in enumerate(tqdm(dataset_source)):
                img, label = img.cuda(),label.long().cuda()

                textual_domain_embedding = resnet101_texture_model.img_encoder(img)

                # textual source prototype -- whole img
                sources_domain_embeddings_textual[n_source] += textual_domain_embedding.cpu().numpy()[0]

            len_source = len(dataset_source.dataset)
            sources_domain_embeddings_textual[n_source] = normalize(sources_domain_embeddings_textual[n_source] / len_source)

    correct_mean = 0
    correct_visual_domain_embedding = 0
    correct_texture_domain_embedding = 0

    softmax = nn.Softmax(dim=-1)

    print('Evaluation on the Target domain - %s' % (args.target))
    with torch.no_grad():
        for it, (img, label) in enumerate(tqdm(dataset_test_target)):

            img, label = img.cuda(), label.long().cuda()

            # logits obtained with model trained with source1
            feat1 = G1(img)
            out1 = softmax(C1(feat1))

            # logits obtained with model trained with source2
            feat2 = G2(img)
            out2 = softmax(C2(feat2))

            # logits obtained with model trained with source3
            feat3 = G3(img)
            out3 = softmax(C3(feat3))

            # textual target embedding -- whole img
            target_domain_embedding_textual = resnet101_texture_model.img_encoder(img)


            pred_mean = ((out1 +out2 + out3) / 3).data.max(1)[1]
            correct_mean += pred_mean.eq(label.data).cpu().sum()


            ####### textual
            b = normalize(target_domain_embedding_textual[0].cpu().numpy())
            a1 = sources_domain_embeddings_textual[0]
            a2 = sources_domain_embeddings_textual[1]
            a3 = sources_domain_embeddings_textual[2]
            w1_text = cosine_sim(a1,b)
            w2_text = cosine_sim(a2,b)
            w3_text = cosine_sim(a3,b)
            pred_only_texture_domain_embedding = ((w1_text * out1 + w2_text * out2 + w3_text * out3) / (w1_text + w2_text + w3_text)).data.max(1)[1]
            correct_texture_domain_embedding += pred_only_texture_domain_embedding.eq(label.data).cpu().sum()

    acc_mean= float(correct_mean.item() / len(dataset_test_target.dataset)) * 100
    print('Accuracy mean: %.2f '% (acc_mean))

    acc_only_texture_domain_embedding = float(correct_texture_domain_embedding.item() / len(dataset_test_target.dataset)) * 100
    print('Accuracy text_domain_embedding: %.2f ' % (acc_only_texture_domain_embedding))



if __name__ == '__main__':
    main()





