import torch
import torch.nn as nn
import torch.nn.functional as F
#Conditional Variational Autoencoder
#Compresses dates into a continuous latent space distribution, then samples and decodes back using input conditions.
#Encoder: Maps input digits + conditions to a latent distribution (mu, logvar).
#Reparameterization: Samples from the distribution using z = mu + std * epsilon.
#Decoder: Reconstructs digits from latent code + conditions.
#Loss: Sum of reconstruction loss + KL divergence (to keep latent space smooth).
class VAEModel(nn.Module):
    def __init__(self, vocab_size, out_vocab_size, max_len=12, latent_dim=64, embed_dim=64, hidden_dim=256):
        super().__init__()
        self.max_len = max_len
        self.out_vocab_size = out_vocab_size
        self.cond_embed = nn.Embedding(vocab_size, embed_dim)
        
        # Encoder
        # ENCODER: Compresses (Flattened Date One-Hot + Condition Embeddings) -> Hidden Representation
        self.encoder = nn.Sequential(
            nn.Linear(max_len * out_vocab_size + embed_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)#Predicts Mean (mu) of latent space
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)#Predicts Log-Variance of latent space
        
        # Decoder
        # DECODER: Reconstructs Flattened Date One-Hot from (Latent Sample z + Condition Embeddings)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + embed_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, max_len * out_vocab_size)
        )
    #Flattens the real date representation, concatenates it with the condition embeddings, passes it through self.encoder, and outputs two parameter vectors: mu and logvar.
    def encode(self, conditions, x_onehot):
        b = conditions.size(0)
        c_emb = self.cond_embed(conditions).view(b, -1)
        x_flat = x_onehot.view(b, -1)
        
        enc_in = torch.cat([x_flat, c_emb], dim=1)
        h = self.encoder(enc_in)
        return self.fc_mu(h), self.fc_logvar(h)
    #Samples from the distribution using z = mu + std * epsilon.
    #This enables backpropagation through the random sampling step.
    #Samples a random gaussian vector, This makes the random sampling step backpropagation-friendly
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    #Reconstructs the date from the latent space and condition embeddings.
    #Merges the latent sample z with condition embeddings and reconstructs the shape (batch, max_len, out_vocab_size).
    def decode(self, conditions, z):
        b = conditions.size(0)
        c_emb = self.cond_embed(conditions).view(b, -1)
        
        dec_in = torch.cat([z, c_emb], dim=1)
        out = self.decoder(dec_in)
        return out.view(b, self.max_len, self.out_vocab_size)
    #Encodes, samples, and decodes in one go.
    #Encodes real dates, reparameterizes to get latent vector z, and decodes to reconstruct logits.
    #Returns reconstructed logits, mu, and logvar (for VAE loss computation).
    def forward(self, conditions, x_onehot):
        mu, logvar = self.encode(conditions, x_onehot)
        z = self.reparameterize(mu, logvar)
        recon_logits = self.decode(conditions, z)
        return recon_logits, mu, logvar
    #Generates dates by decoding random latent vectors (or provided ones) conditioned on given categories.
    #If no latent vector is provided, it samples random Gaussian vectors.
    def generate(self, conditions, z=None):
        if z is None:
            b = conditions.size(0)
            z = torch.randn(b, self.fc_mu.out_features, device=conditions.device)
        logits = self.decode(conditions, z)
        return logits.argmax(dim=-1)
# Inference mode. Samples a noise vector z from standard normal distribution, pairs it with conditions, decodes, and calls logits.argmax(dim=-1) to output characters.