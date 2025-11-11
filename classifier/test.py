import time
from facenet_pytorch import MTCNN, InceptionResnetV1

from PIL import Image
import os
import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

import dataloader
import json

IMAGE_SIZE = 128
NB_IMAGES = 10

HAS_CUDA = torch.cuda.is_available()
print("cuda compatible :", HAS_CUDA)
device = torch.device("cuda" if HAS_CUDA else "cpu")


print("loading resnet")
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device) #permet d'extraire des embeddings des visages MAIS du coup est ce qu'on utilise ca ou on fait notre propre truc ?

data_dir = "classifier/dataset/LFW_small/"
train_dataset, test_dataset, nb_classes = dataloader.make_dataset(data_dir, NB_IMAGES, IMAGE_SIZE)


#Definition du modèle
class Encoder(nn.Module):
    def __init__(self, embedding_size, nb_classes):
        super(Encoder, self).__init__()

        self.embedding_size = embedding_size
        self.nb_classes = nb_classes

        """ self.encoder = nn.Sequential(
            nn.Linear(IMAGE_SIZE * IMAGE_SIZE * 3 , embedding_size*8),
            nn.ReLU(),
            nn.Linear(embedding_size*8, embedding_size*4),
            nn.ReLU(),
            nn.Linear(embedding_size*4, embedding_size*2),
            nn.ReLU(),
            nn.Linear(embedding_size*2, embedding_size),
        ) """

        self.encoder = nn.Sequential(
            nn.Linear(embedding_size, embedding_size*2),
            nn.ReLU(),
            nn.Linear(embedding_size*2, embedding_size),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(embedding_size, nb_classes)

    def forward(self,x):
        #encoded = self.encoder(x)
        classif = self.classifier(x)
        return classif


#Encodeur et classification

def make_experiment(optimizer_name, learning_rate, batch_size, epochs, embedding_size, exp_id):
    experiment_name = f"exp_{exp_id}_{optimizer_name}_lr{learning_rate}_bs{batch_size}_emb{embedding_size}"

    print(f"\n===== EXPÉRIENCE {exp_id} : {experiment_name} =====")

    #dossier de sortie
    result_dir = os.path.join("results", experiment_name)
    os.makedirs(result_dir, exist_ok=True)

    #datasets
    #train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    train_embeddings = dataloader.make_embeddings(train_dataset, resnet, device)
    print(train_embeddings.tensors[0][0][:5])

    test_embeddings = dataloader.make_embeddings(test_dataset, resnet, device)

    train_loader = DataLoader(train_embeddings, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_embeddings, batch_size=batch_size, shuffle=True)

    #creation du modèle
    model = Encoder(embedding_size, nb_classes)
    model.to(device)

    crossEntropyLoss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001) #connecte le modèle et l'optimizer

    outputs = []
    losses = []

    model.train()

    #entrainement du modèle
    for epoch in range(epochs):
        for data, labels in train_loader:


            #images = images.view(-1, IMAGE_SIZE * IMAGE_SIZE * 3).to(device) #faut applatir l'image 
            data = data.to(device)

            labels = labels.to(device) # transfere sur le GPU

            guess = model(data) # forward dans le modèle
            expected = labels # les vraies réponses

            loss = crossEntropyLoss(guess, expected)

            optimizer.zero_grad()
            loss.backward() # backpropagation
            optimizer.step()

            losses.append(loss.item())

            outputs.append((epoch, data, guess))

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    
    
    #test du modèle

    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    with torch.no_grad():  # desactive le calcul des gradients, plus rapide pour le test
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)

            outputs = model(data)
            loss = crossEntropyLoss(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = correct / total
    test_loss /= len(test_loader)

    print(f"Test loss : {test_loss:.4f}")
    print(f"Test accuracy : {test_accuracy:.4f}")

    metrics = {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "params": {
            "optimizer": optimizer_name,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs
        }
    }

    with open(os.path.join(result_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    plt.figure()
    plt.plot(losses, label="Loss")
    plt.title(f"{experiment_name}")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(os.path.join(result_dir, "loss_plot.png"))
    plt.close()

    torch.save(model.state_dict(), os.path.join(result_dir, "weights.pth"))
    print(f"resultats enregistre dans : {result_dir}\n")




""" make_experiment(
    optimizer_name="Adam",
    learning_rate=0.001,
    batch_size=32,
    epochs=200,
    embedding_size=512, #forcement 512 car on utilise inception resnet
    exp_id=1
)
 """








