# Implementation of Variational Auto Encoder using CNN and Transpose CNN

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Data Set, Transformation and Data Loader
transform = transforms.ToTensor()
mnist_data = datasets.MNIST(root='/home/idrbt-06/Desktop/PY_TORCH/Auto_Encoder/Data', train=True, download=True, transform=transform)
data_loader = torch.utils.data.DataLoader(dataset=mnist_data, batch_size=10, shuffle=True)

# Preparation of the Model for Variational Auto Encoder

class VariationalAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=25,kernel_size=3,stride=2,padding=1,bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=25,out_channels=50,kernel_size=3,stride=2,padding=1,bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=50,out_channels=100,kernel_size=7,bias=False),
        )
        
        self.latent_space_for_mean = nn.Linear(100,20)
        self.latent_space_for_log_variance = nn.Linear(100,20)
        self.dimension_conversion = nn.Linear(20,100)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=100,out_channels=50,kernel_size=7,bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=50,out_channels=25,kernel_size=3,stride=2,padding=1,output_padding=1,bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=25,out_channels=1,kernel_size=3,stride=2,padding=1,output_padding=1,bias=False),
            nn.Sigmoid()
        )
    
    def encoding(self, x):
        encoded_output = self.encoder(x)
        return encoded_output
    
    def mean_log_variance_calculation(self, x):
        mean_vector = self.latent_space_for_mean(x)
        #print(mean_vector.shape)
        log_variance_vector = self.latent_space_for_log_variance(x)
        #print(log_variance_vector.shape)
        return mean_vector, log_variance_vector
    
    def standard_deviation_calculation(self, x):
        variance_vector = torch.exp(x)
        standard_deviation_vector = torch.sqrt(variance_vector)
        return standard_deviation_vector
    
    def reparameterization(self, mean, sigma):
        epsilon = torch.randn(sigma.shape)
        z = mean + epsilon*sigma
        return z
    
    def decoding(self, x):
        x1 = self.dimension_conversion(x)
        x2 = x1.view(-1,100,1,1)
        #print(x2.shape)
        output = self.decoder(x2)
        return output
    
    def forward(self,x):
        #print(x.shape)
        encoded_output = self.encoding(x)
        #print(encoded_output.shape)
        encoded_output_flatten = encoded_output.flatten(start_dim=1)
        #print(encoded_output_flatten.shape)
        mean_vector, log_variance_vector = self.mean_log_variance_calculation(encoded_output_flatten)
        standard_deviation_vector = self.standard_deviation_calculation(log_variance_vector)
        z = self.reparameterization(mean_vector, standard_deviation_vector)
        #print(z.shape)
        output = self.decoding(z)
        #print(output.shape)
        return output, mean_vector, log_variance_vector


model = VariationalAutoencoder()

# Loss Calculation for Variational Autoencoder
class Loss_for_Variational_Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input, output, mean_vector, log_variance_vector):

        reconstruction_loss = F.binary_cross_entropy(output, input, reduction='sum')

        kld_first_term = torch.sum(torch.exp(log_variance_vector), dim = 1)
        kld_second_term = torch.sum((mean_vector**2), dim = 1)
        kld_third_term_addition_fourth_term = torch.sum((log_variance_vector+1), dim = 1)
        KL_Divergence_loss = 0.5*(kld_first_term + kld_second_term - kld_third_term_addition_fourth_term)
        kld = torch.sum(KL_Divergence_loss)
        
        total_loss = reconstruction_loss + kld
        return total_loss


criterion = Loss_for_Variational_Autoencoder()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training of Variational Autoencoder
num_epochs = 502
original_and_reconstructed_images = []

for epoch in range(0, num_epochs, 1):
    for (images,labels) in data_loader:

        output, mean_vector, log_variance_vector = model(images)
        loss = criterion(images, output, mean_vector, log_variance_vector)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch:{epoch+1}, Loss:{loss.item():.4f}')
    original_and_reconstructed_images.append((epoch+1, images, output),)


# Display of Original and Reconstructed Images
# Here I have considering every 50 epoch
for k in range(0, num_epochs, 50):
    plt.figure(k)
    (a, b, c) = original_and_reconstructed_images[k]
    print('This is a=', a)
    print('This is the size of b=', b.shape)
    print('This is the size of c=', c.shape)
    original = b
    reconstructed = c
    original = original.detach().numpy()
    reconstructed = reconstructed.detach().numpy()
    for i in range(0, 10, 1):
        plt.subplot(2, 10, i+1)
        plt.imshow(original[i][0])
        plt.subplot(2, 10, 10+i+1)
        plt.imshow(reconstructed[i][0])
plt.show()

