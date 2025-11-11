from torch import nn, optim
import torch
import dataloader
from torch.utils.data import DataLoader
from torchvision import models
from facenet_pytorch import InceptionResnetV1
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image
import matplotlib.pyplot as plt

import utils
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


IMAGE_SIZE = 128
NB_IMAGES = 10
EMBEDDING_SIZE = 5
BATCH_SIZE = 32

EPOCHS = 500

train_dataset, test_dataset, nb_classes = dataloader.make_dataset("classifier/dataset/LFW_small", NB_IMAGES, IMAGE_SIZE)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)



#------------------------------definition des modeles------------------------------------------

class Embedder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.embed = nn.Linear(IMAGE_SIZE * IMAGE_SIZE * 3, EMBEDDING_SIZE)

    def forward(self,x):
        res = self.embed(x)
        res = res / res.norm(dim= 1, keepdim=True)
        return self.embed(x)

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fc = nn.Linear(EMBEDDING_SIZE, num_classes)
    def forward(self, x):
        return self.fc(x)






#------------------------------entrainement embedder------------------------------------------

faceEmbedder = Embedder().to(device)
triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
optimizer = optim.Adam(faceEmbedder.parameters(), lr=0.001)

final_embedder_loss = 0.0

faceEmbedder.train()
for epoch in range(EPOCHS):

    total_loss = 0.0

    for images, labels in train_loader:
        images = torch.flatten(images, 1)

        images = images.to(device)
        labels = labels.to(device)

        output = faceEmbedder(images)
        anchor_idx, positive_idx, negative_idx = utils.get_triplets(labels)

        anchors = output[anchor_idx]
        positives = output[positive_idx]
        negatives = output[negative_idx]

        loss = triplet_loss(anchors,positives, negatives)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        

        total_loss += loss.item()

    total_loss = total_loss / len(train_loader)

    #print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
    final_embedder_loss = total_loss





print("Embedder final loss : ", final_embedder_loss)








#------------------------------entrainement classifieur------------------------------------------

classifier = Classifier(nb_classes).to(device)
crossLoss = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.001)

final_classifier_loss = [0.0,0.0]

faceEmbedder.eval()
for epoch in range(EPOCHS):
    total_loss = 0.0
    classifier.train()
    for images, labels in train_loader:
        images = torch.flatten(images, 1)

        images = images.to(device)
        labels = labels.to(device)

        embeded = faceEmbedder(images)

        output = classifier(embeded)
        loss = crossLoss(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()


    #test avec le set de validation
    classifier.eval()
    total_eval_loss = 0.0
    for images, labels in test_loader:
        images = torch.flatten(images, 1)
        images = images.to(device)
        labels = labels.to(device)

        embeded = faceEmbedder(images)

        output = classifier(embeded)
        loss = crossLoss(output, labels)

        #metrics
        total_eval_loss += loss.item()


    total_loss = total_loss / len(train_loader)
    total_eval_loss = total_eval_loss / len(test_loader)
    #print(f"Epoch {epoch+1}, training loss: {total_loss:.4f}, test loss: {total_eval_loss:.4f}")

    final_classifier_loss[0] = total_loss
    final_classifier_loss[1] = total_eval_loss



print("Classifier final train loss : ", final_classifier_loss[0], " test loss : ", final_classifier_loss[1])


#------------------------------analyse des resultats------------------------------------------

# analyse des embeddings
faceEmbedder.eval()
embeddings = []
with torch.no_grad():
    for images, labels in train_loader:
        #print(labels)
        images = torch.flatten(images, 1)

        images = images.to(device)
        labels = labels.to(device)

        output = faceEmbedder(images)

        for i in range(output.size(0)):
            embeddings.append([output[i].cpu().tolist(), labels[i].cpu().item()])

def plot_embeddings():
    embeddings_np = np.array([e[0] for e in embeddings])
    labels_np = np.array([e[1] for e in embeddings])

    #reduction en 2D
    pca = PCA(n_components=2) 
    reduced_embeddings = pca.fit_transform(embeddings_np)

    plt.figure(figsize=(10, 10))
    for label in np.unique(labels_np):
        idx = labels_np == label
        plt.scatter(reduced_embeddings[idx, 0], reduced_embeddings[idx, 1], label=f"classe {label}")

    plt.title("embeddings")
    plt.legend()
    plt.grid(True)
    plt.show()



plot_embeddings()


#matrice de confusion
def plot_confusion_matrix():
    classifier.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = torch.flatten(images, 1)
            images = images.to(device)
            labels = labels.to(device)

            embeded = faceEmbedder(images)
            output = classifier(embeded)
            predictions = torch.argmax(output, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    cm = confusion_matrix(all_labels, all_predictions, labels=range(nb_classes))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f"classe {i}" for i in range(nb_classes)])
    disp.plot()
    plt.title("Matrice de confusion")
    plt.show()


plot_confusion_matrix()
