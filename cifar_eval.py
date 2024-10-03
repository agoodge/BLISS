import os
import torch
import data_utils
import faiss
import argparse
from tqdm import tqdm
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score
from imagenet_utils.dictionary import dictionary
from clip.simple_tokenizer import SimpleTokenizer as clip_tokenizer
import warnings
warnings.filterwarnings("ignore")

# tokenize sentences
def tokenize_for_clip(batch_sentences, tokenizer):
    default_length = 77
    sot_token = tokenizer.encoder['<|startoftext|>']
    eot_token = tokenizer.encoder['<|endoftext|>']
    tokenized_list = []
    for sentence in batch_sentences:
        text_tokens = [sot_token] + tokenizer.encode(sentence) + [eot_token]
        tokenized = torch.zeros((default_length), dtype=torch.long)
        tokenized[:len(text_tokens)] = torch.tensor(text_tokens)
        tokenized_list. append(tokenized)
    tokenized_list = torch.stack(tokenized_list)
    return tokenized_list

# encode images from loader with clip model
def image_encoder(clip_model, device, loader):
    with torch.no_grad():
        image_features = []
        for idx, image in enumerate(tqdm(loader)):
            image_feature = clip_model.encode_image(image.to(device)).float()
            image_features.append(image_feature.detach().cpu().numpy())
        image_features = np.vstack(image_features)
    return image_features

# encode text in labels with clip_model
def text_encoder(clip_model, device, labels):
    with torch.no_grad():
        cliptokenizer = clip_tokenizer()
        seen_descriptions = [f"This is a photo of a {label}" for label in labels]
        seen_descriptions_tokens = tokenize_for_clip(seen_descriptions, cliptokenizer)
        text_features = clip_model.encode_text(seen_descriptions_tokens.to(device)).float()
    return text_features.detach().cpu().numpy()

# return cosine similarity between normalized features
def sim(image_features, text_features):
    return image_features @ text_features.T

# similarity to normal text labels from dataset
def internal_class_score(normal_val_image_features, test_image_features, dataset_text_features):
    val_dataset_text_distances = sim(normal_val_image_features,dataset_text_features)                
    dataset_text_mu = np.mean(val_dataset_text_distances,axis=0)
    dataset_text_std = np.std(val_dataset_text_distances,axis=0) + 1e-8
    test_dataset_text_distances = sim(test_image_features,dataset_text_features)
    score = -(test_dataset_text_distances - dataset_text_mu)/dataset_text_std
    return score

# similarity to text labels from dictionary
def external_test_score(normal_val_image_features, test_image_features, dictionary_text_features, topk_dictionary_distances, topk_dictionary_labels):
    val_dictionary_text_distances = sim(normal_val_image_features,dictionary_text_features)
    dictionary_text_mu = np.mean(val_dictionary_text_distances, axis = 0)
    dictionary_text_std = np.std(val_dictionary_text_distances, axis = 0) + 1e-8
    dictionary_score = (topk_dictionary_distances - dictionary_text_mu[topk_dictionary_labels])/dictionary_text_std[topk_dictionary_labels]
    dictionary_score = dictionary_score.reshape(len(dictionary_score),-1)
    score = dictionary_score.mean(axis=1)
    return score

# normalize features to unit sphere
def norm(features):
    features /= np.linalg.norm(features,axis=1)[:, None]
    return features

def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip_model = torch.jit.load(os.path.join('./trained_models', "{}.pt".format('ViT-B-16')))#.to(device).eval()

    print("Encoding training images...")
    val_loader = data_utils.data_loader(args.dataset, train = True)
    val_image_features = image_encoder(clip_model, device, val_loader)
    val_image_features = norm(val_image_features)
    val_image_targets = np.array(val_loader.dataset.targets)
    
    print("Encoding test images...")
    test_loader = data_utils.data_loader(args.dataset, train = False)
    test_image_features = image_encoder(clip_model, device, test_loader)
    test_image_features = norm(test_image_features)
    test_image_targets = np.array(test_loader.dataset.targets)

    print("Encoding text...")
    global dictionary
    dictionary_list = list(dictionary.values())
    dictionary = []
    for label in tqdm(range(len(dictionary_list))):
        for item in dictionary_list[label].split(','):
            dictionary.append(item)
    dictionary_text_features = text_encoder(clip_model, device, dictionary)
    dictionary_text_features = norm(dictionary_text_features)

    dataset_text_features = text_encoder(clip_model, device, val_loader.dataset.classes)
    dataset_text_features = norm(dataset_text_features)

    # find the topk most similar words in dictionary_text_features to each image in test_image_features
   
    test_dictionary_text_distances = sim(test_image_features, dictionary_text_features)
    topk_dictionary_distances = np.sort(test_dictionary_text_distances,axis=1)[:,-args.topk:].squeeze()
    topk_dictionary_labels = np.argsort(test_dictionary_text_distances,axis=1)[:,-args.topk:].squeeze()

    if args.dataset == 'cifar100':
        assert args.setup == '6'

    if args.setup =='1':
        splits = []
        for i in val_loader.dataset.classes:
            splits.append([val_loader.dataset.class_to_idx[i]])
    elif args.setup == '6':
        splits = val_loader.dataset.zoc_splits
    elif args.setup =='9':
        splits = []
        for i in val_loader.dataset.classes:
            temp = set(val_loader.dataset.classes) - set([i])
            splits.append([val_loader.dataset.class_to_idx[i] for i in temp])

    split_auc_list = []

    for normal_target_set in splits:
        all_class_scores = []
        # for each class in the set of normal classes
        for single_class in normal_target_set:
            
            #only normal images belonging to digit class
            normal_val_image_features = val_image_features[val_image_targets == single_class]

            single_class_score = np.zeros(len(test_image_features))

            # similarity to normal text labels from datasets)
            i_scores = internal_class_score(normal_val_image_features, test_image_features, dataset_text_features[single_class])
            single_class_score += i_scores

            # similarity to text labels from dictionary
            e_scores = external_test_score(normal_val_image_features, test_image_features, dictionary_text_features, topk_dictionary_distances, topk_dictionary_labels)
            single_class_score += args.lam*e_scores
            
            all_class_scores.append(single_class_score)
        scores = np.array(all_class_scores).min(axis=0)
        test_normal_targets = np.array([0 if i in normal_target_set else 1 for i in test_loader.dataset.targets])
        auc = 100*roc_auc_score(test_normal_targets, scores)
        split_auc_list.append(auc)

    split_auc_list = np.array(split_auc_list)
    print("Mean AUROC:",np.round(np.mean(split_auc_list,axis=0),1))
    print("Std Deviation:", np.round(np.std(split_auc_list,axis=0),1))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default = 'cifar100', help = 'Choice of dataset between cifar10 and cifa100')
    parser.add_argument("--topk", type = str, default = 10, help = 'Number of dictionary words to use in external text score')
    parser.add_argument("--lam", type = float, default = 0.5, help = 'Value of the hyper-parameter weight of external text score')
    parser.add_argument("--setup", type = str, default ="6", help = 'Choice between 1, 6 or 9 normal class setup')
    args = parser.parse_args()

    main(args)
