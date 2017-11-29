import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time

# --- 1. Configuration et Détection du GPU ---
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Utilisation du GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA non disponible, utilisation du CPU.")
    return device

device = get_device()

# --- 2. Définition du Modèle (CNN Simple) ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 3 conv layers, 3 max pool layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), # 32x32x3 -> 32x32x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 32x32x32 -> 16x16x32
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 16x16x32 -> 16x16x64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 16x16x64 -> 8x8x64
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 8x8x64 -> 8x8x128
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 8x8x128 -> 4x4x128
        )
        # Flatten et couches fully connected
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512), # 4x4x128 = 2048
            nn.ReLU(),
            nn.Dropout(0.5), # Regularisation
            nn.Linear(512, 10) # 10 classes pour CIFAR-10
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) # Aplatir pour les couches linéaires
        x = self.classifier(x)
        return x

# --- 3. Chargement et Préparation des Données CIFAR-10 ---
# Transformations pour les images : Conversion en tenseur et Normalisation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalisation pour les 3 canaux
])

# Téléchargement et chargement des jeux de données d'entraînement et de test
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0) # num_workers peut être ajusté

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'chicken', 'frog', 'horse', 'ship', 'truck')

# --- 4. Initialisation du Modèle, de la Fonction de Perte et de l'Optimiseur ---
model = SimpleCNN().to(device) # Envoyer le modèle sur le GPU si disponible
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- 5. Boucle d'Entraînement ---
num_epochs = 10 # Nombre d'époques pour le training

print(f"\nDébut de l'entraînement sur {num_epochs} époques...")
start_time = time.time()

for epoch in range(num_epochs):
    model.train() # Mettre le modèle en mode entraînement
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # Récupérer les inputs et les labels et les envoyer sur le GPU
        inputs, labels = data[0].to(device), data[1].to(device)

        # Zéro-grad les paramètres de l'optimiseur
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass + Optimisation
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99: # Afficher toutes les 100 mini-batchs
            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(trainloader)}], Loss: {running_loss / 100:.3f}")
            running_loss = 0.0

end_time = time.time()
print("\nEntraînement terminé.")
print(f"Temps total d'entraînement : {end_time - start_time:.2f} secondes")

# --- 6. Évaluation du Modèle sur le Jeu de Test ---
print("\nDébut de l'évaluation sur le jeu de test...")
model.eval() # Mettre le modèle en mode évaluation
correct = 0
total = 0
with torch.no_grad(): # Désactiver le calcul des gradients pour l'évaluation
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Précision du réseau sur les 10000 images de test: {100 * correct / total:.2f}%')