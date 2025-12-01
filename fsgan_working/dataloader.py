import os
from PIL import Image
import torch
from torch.utils.data import TensorDataset
from facenet_pytorch import MTCNN, InceptionResnetV1
from torch.utils.data import DataLoader, TensorDataset
import numpy as np



def make_dataset(data_dir, nb_images, image_size, trainsplit, crop_faces = True):
    faces = []  # cropped
    labels = []  # index des personnes
    name_to_idx = {}  # mapping nom -> index

    mtcnn = MTCNN(image_size=image_size, margin=0) #decoupe les images pour avoir que les visages et transforme en tensor
    print("Building dataset \n")
    for idx, person in enumerate(sorted(os.listdir(data_dir))):
        if idx >= 500:
            pass#break

        person_dir = os.path.join(data_dir, person)
        name_to_idx[person] = idx

        maxit = nb_images  # nb max d'images par personne
        it = 0
        for fname in sorted(os.listdir(person_dir)):
            if it >= maxit:
                break
            it += 1
            fpath = os.path.join(person_dir, fname)

            try:
                img = Image.open(fpath)
                if crop_faces:
                    face = mtcnn(img)
                    if face is None:
                        print(f"face is none {fpath}")
                        continue
                    faces.append(face)
                else:
                    img = img.resize((image_size, image_size))
                    faces.append(torch.tensor(np.array(img)).permute(2, 0, 1))
                labels.append(idx)
            except Exception as e:
                print(f"Error processing {fpath}: {e}")
                continue

    complete_faces = torch.stack(faces)
    complete_labels = torch.tensor(labels, dtype=torch.long)

    complete_faces = complete_faces.float()
    if not crop_faces : complete_faces = (complete_faces/127.5) -1.0  #on normalise
    #complete_faces = complete_faces / 255.0

    nb_classes = len(name_to_idx.keys())
    
    #split 70 30, equilibre
    train_faces_list, train_labels_list = [], []
    test_faces_list, test_labels_list = [], []

    for cls in range(nb_classes):
        cls_indices = (complete_labels == cls).nonzero(as_tuple=True)[0]
        cls_faces = complete_faces[cls_indices]
        cls_labels = complete_labels[cls_indices]

        split_idx = int(len(cls_faces) * trainsplit)
        train_faces_list.append(cls_faces[:split_idx])
        train_labels_list.append(cls_labels[:split_idx])
        test_faces_list.append(cls_faces[split_idx:])
        test_labels_list.append(cls_labels[split_idx:])

    train_faces = torch.cat(train_faces_list)
    train_labels = torch.cat(train_labels_list)
    test_faces = torch.cat(test_faces_list)
    test_labels = torch.cat(test_labels_list)


    #print("faces train set size :", train_faces.size(0))
    #print("faces test set size :", test_faces.size(0))
    #print("labels train set size:", train_labels.size(0))
    #print("labels test set size:", test_labels.size(0))

    train_dataset = TensorDataset(train_faces, train_labels)
    test_dataset = TensorDataset(test_faces, test_labels)

    return train_dataset, test_dataset, nb_classes



def make_embeddings(dataset, resnet ,device):
    print("generating embeddings \n")
    embeddings = []
    labels = []

    resnet.eval()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    with torch.no_grad():
        for batch in dataloader:
            imgs, batch_labels = batch
            imgs = imgs.to(device)
            batch_embeddings = resnet(imgs)
            batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=1, keepdim=True)
            embeddings.append(batch_embeddings)
            labels.append(batch_labels.to(device))

    embeddings_tensor = torch.cat(embeddings, dim=0)
    labels_tensor = torch.cat(labels, dim=0)

    print("Embeddings shape:", embeddings_tensor.shape)
    print("Labels shape:", labels_tensor.shape)

    dataset = TensorDataset(embeddings_tensor, labels_tensor)

    return dataset