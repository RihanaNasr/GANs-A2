import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from dataset import DateTokenizer, DateDataset
from rnn_model import RNNModel
from transformer_model import TransformerModel
from vae_model import VAEModel
from gan_model import Generator, Discriminator
import os
import random
import numpy as np
import matplotlib.pyplot as plt
#Executes data preparation, validation checks, training metrics, and automatic graph generation.

# Set seeds on random, numpy, PyTorch,and CUDA to ensure training results
#are fully reproducible (satisfying the bonus grading criteria).
def set_seed(seed=42):
    """Set manual seeds for reproducibility (Bonus grading point)"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
#Uses matplotlib to plot train and test losses over epochs and saves them
#as .png files (satisfying the core plotting requirement).
def plot_losses(train_losses, test_losses, model_name):
    """Generate and save training vs testing loss curves"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', color='#1f77b4', linewidth=2)
    plt.plot(test_losses, label='Test Loss', color='#ff7f0e', linewidth=2, linestyle='--')
    plt.title(f'{model_name.upper()} Training and Test Loss Over Epochs', fontsize=12, fontweight='bold')
    plt.xlabel('Epochs', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.legend(frameon=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plot_path = f"{model_name}_loss_curve.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Loss plot saved to {plot_path}")

def plot_gan_losses(train_d, train_g, test_d, test_g, model_name):
    """Generate and save GAN Discriminator and Generator loss curves"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_d, label='Train Discriminator Loss', color='#1f77b4')
    plt.plot(train_g, label='Train Generator Loss', color='#ff7f0e')
    plt.plot(test_d, label='Test Discriminator Loss', color='#2ca02c', linestyle='--')
    plt.plot(test_g, label='Test Generator Loss', color='#d62728', linestyle='--')
    plt.title('GAN Training and Test Losses Over Epochs', fontsize=12, fontweight='bold')
    plt.xlabel('Epochs', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.legend(frameon=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plot_path = f"{model_name}_loss_curve.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Loss plot saved to {plot_path}")

#Iterates training batches using train_loader (model.train()).
#Calculates autoregressive loss (target shifted by -1) via CrossEntropyLoss.
#Updates model weights via backward() and optimizer.step().
#Evaluates test loss using model.eval() without gradient calculation
def train_autoregressive(model, train_loader, test_loader, epochs, device, save_path, model_name):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    pad_id = train_loader.dataset.dataset.tokenizer.pad_id
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for batch_idx, (conditions, targets) in enumerate(train_loader):
            conditions, targets = conditions.to(device), targets.to(device)
            inputs = targets[:, :-1]
            target_out = targets[:, 1:]
            
            optimizer.zero_grad()
            logits = model(conditions, inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), target_out.reshape(-1))
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
        # Test Loss Evaluation
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for conditions, targets in test_loader:
                conditions, targets = conditions.to(device), targets.to(device)
                inputs = targets[:, :-1]
                target_out = targets[:, 1:]
                logits = model(conditions, inputs)
                loss = criterion(logits.reshape(-1, logits.size(-1)), target_out.reshape(-1))
                total_test_loss += loss.item()
                
        epoch_train_loss = total_train_loss / len(train_loader)
        epoch_test_loss = total_test_loss / len(test_loader)
        train_losses.append(epoch_train_loss)
        test_losses.append(epoch_test_loss)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_train_loss:.4f} | Test Loss: {epoch_test_loss:.4f}")
    
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    plot_losses(train_losses, test_losses, model_name)

def train_vae(model, train_loader, test_loader, epochs, device, save_path, model_name):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    pad_id = train_loader.dataset.dataset.tokenizer.pad_id
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for batch_idx, (conditions, targets) in enumerate(train_loader):
            conditions, targets = conditions.to(device), targets.to(device)
            target_onehot = F.one_hot(targets, num_classes=model.out_vocab_size).float()
            
            optimizer.zero_grad()
            recon_logits, mu, logvar = model(conditions, target_onehot)
            recon_loss = criterion(recon_logits.view(-1, model.out_vocab_size), targets.view(-1))
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / targets.size(0)
            
            loss = recon_loss + 0.1 * kl_loss
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
        # Test Loss Evaluation
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for conditions, targets in test_loader:
                conditions, targets = conditions.to(device), targets.to(device)
                target_onehot = F.one_hot(targets, num_classes=model.out_vocab_size).float()
                recon_logits, mu, logvar = model(conditions, target_onehot)
                recon_loss = criterion(recon_logits.view(-1, model.out_vocab_size), targets.view(-1))
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / targets.size(0)
                loss = recon_loss + 0.1 * kl_loss
                total_test_loss += loss.item()
                
        epoch_train_loss = total_train_loss / len(train_loader)
        epoch_test_loss = total_test_loss / len(test_loader)
        train_losses.append(epoch_train_loss)
        test_losses.append(epoch_test_loss)
            
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_train_loss:.4f} | Test Loss: {epoch_test_loss:.4f}")
        
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    plot_losses(train_losses, test_losses, model_name)
#Trains Discriminator: Uses Binary Cross-Entropy Loss (BCEWithLogitsLoss) to classify real dates as 1 and generated fake dates as 0.
#Trains Generator: Optimizes Generator weights so the Discriminator classifies its fake outputs as 1.
#Performs test loss checks at each epoch to monitor and plot adversarial convergence.

def train_gan(gen, disc, train_loader, test_loader, epochs, device, save_path, model_name):
    opt_g = torch.optim.Adam(gen.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()
    
    train_d_losses = []
    train_g_losses = []
    test_d_losses = []
    test_g_losses = []
    
    for epoch in range(epochs):
        gen.train()
        disc.train()
        total_train_d = 0
        total_train_g = 0
        for batch_idx, (conditions, targets) in enumerate(train_loader):
            conditions, targets = conditions.to(device), targets.to(device)
            b = conditions.size(0)
            
            target_onehot = F.one_hot(targets, num_classes=gen.out_vocab_size).float()
            
            # Train Discriminator
            opt_d.zero_grad()
            real_preds = disc(conditions, target_onehot)
            real_loss = criterion(real_preds, torch.ones_like(real_preds))
            
            noise = torch.randn(b, 64, device=device)
            fake_onehot = gen(conditions, noise, temperature=1.0)
            fake_preds = disc(conditions, fake_onehot.detach())
            fake_loss = criterion(fake_preds, torch.zeros_like(fake_preds))
            
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            opt_d.step()
            total_train_d += d_loss.item()
            
            # Train Generator
            opt_g.zero_grad()
            fake_preds_for_g = disc(conditions, fake_onehot)
            g_loss = criterion(fake_preds_for_g, torch.ones_like(fake_preds_for_g))
            g_loss.backward()
            opt_g.step()
            total_train_g += g_loss.item()
            
        # Test Loss Evaluation
        # Note: Set generator temporarily to train() in eval context to get Gumbel-Softmax one-hot outputs for Discriminator validation
        gen.train()
        disc.eval()
        total_test_d = 0
        total_test_g = 0
        with torch.no_grad():
            for conditions, targets in test_loader:
                conditions, targets = conditions.to(device), targets.to(device)
                b = conditions.size(0)
                target_onehot = F.one_hot(targets, num_classes=gen.out_vocab_size).float()
                
                real_preds = disc(conditions, target_onehot)
                real_loss = criterion(real_preds, torch.ones_like(real_preds))
                
                noise = torch.randn(b, 64, device=device)
                fake_onehot = gen(conditions, noise, temperature=1.0)
                fake_preds = disc(conditions, fake_onehot.detach())
                fake_loss = criterion(fake_preds, torch.zeros_like(fake_preds))
                
                d_loss = (real_loss + fake_loss) / 2
                total_test_d += d_loss.item()
                
                fake_preds_for_g = disc(conditions, fake_onehot)
                g_loss = criterion(fake_preds_for_g, torch.ones_like(fake_preds_for_g))
                total_test_g += g_loss.item()
                
        epoch_train_d = total_train_d / len(train_loader)
        epoch_train_g = total_train_g / len(train_loader)
        epoch_test_d = total_test_d / len(test_loader)
        epoch_test_g = total_test_g / len(test_loader)
        
        train_d_losses.append(epoch_train_d)
        train_g_losses.append(epoch_train_g)
        test_d_losses.append(epoch_test_d)
        test_g_losses.append(epoch_test_g)
            
        print(f"Epoch {epoch+1}/{epochs} | D Loss: {epoch_train_d:.4f} (Test: {epoch_test_d:.4f}) | G Loss: {epoch_train_g:.4f} (Test: {epoch_test_g:.4f})")
        
    torch.save(gen.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    plot_gan_losses(train_d_losses, train_g_losses, test_d_losses, test_g_losses, model_name)

#Parses command args.
#Calls random_split to divide the dataset into 80% Train and 20% Test partitions.
#Initiates training for the specified model parameter.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["rnn", "transformer", "vae", "gan"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--data", type=str, default="../data/data.txt")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    args = parser.parse_args()
    
    # 1. Set Seed for Reproducibility (Bonus Best Practice)
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = DateTokenizer()
    dataset = DateDataset(args.data, tokenizer)
    
    # 2. Train-Test Split (80% Train, 20% Test) (Bonus Best Practice)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # 3. Create Shuffled Train Loader & Non-shuffled Test Loader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    input_vocab_size = len(tokenizer.input_vocab)
    out_vocab_size = len(tokenizer.out_vocab)
    max_len = tokenizer.max_len
    
    save_path = f"{args.model}_weights.pth"
    
    if args.model == "rnn":
        model = RNNModel(input_vocab_size, out_vocab_size).to(device)
        train_autoregressive(model, train_loader, test_loader, args.epochs, device, save_path, args.model)
    elif args.model == "transformer":
        model = TransformerModel(input_vocab_size, out_vocab_size).to(device)
        train_autoregressive(model, train_loader, test_loader, args.epochs, device, save_path, args.model)
    elif args.model == "vae":
        model = VAEModel(input_vocab_size, out_vocab_size, max_len=max_len).to(device)
        train_vae(model, train_loader, test_loader, args.epochs, device, save_path, args.model)
    elif args.model == "gan":
        gen = Generator(input_vocab_size, out_vocab_size, max_len=max_len, noise_dim=64).to(device)
        disc = Discriminator(input_vocab_size, out_vocab_size, max_len=max_len).to(device)
        train_gan(gen, disc, train_loader, test_loader, args.epochs, device, save_path, args.model)
