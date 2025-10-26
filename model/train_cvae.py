import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm

# Model architecture (same as in app.py)
ELEMENTS = ['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar','K','Ca','Ti','Fe']
EL_TO_IDX = {e:i for i,e in enumerate(ELEMENTS)}
N_ELEM = len(ELEMENTS)
MAX_ATOMS = 64
LATENT_DIM = 128

class SpaceGroupEmbedding(nn.Module):
    def __init__(self, max_sg=230, emb_dim=64):
        super().__init__()
        self.embed = nn.Embedding(max_sg+1, emb_dim)
    
    def forward(self, x):
        return self.embed(x)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.sg_emb = SpaceGroupEmbedding()
        in_dim = 9 + MAX_ATOMS*3 + MAX_ATOMS*N_ELEM
        self.fc1 = nn.Linear(in_dim + 64, 512)
        self.fc_mu = nn.Linear(512, LATENT_DIM)
        self.fc_logvar = nn.Linear(512, LATENT_DIM)
    
    def forward(self, lattice, frac, species_oh, sg_idx):
        x = torch.cat([lattice.view(lattice.size(0), -1), 
                       frac.view(frac.size(0), -1), 
                       species_oh.view(species_oh.size(0), -1)], dim=-1)
        sg_e = self.sg_emb(sg_idx)
        x = torch.cat([x, sg_e], dim=-1)
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.sg_emb = SpaceGroupEmbedding()
        out_dim = 9 + MAX_ATOMS*3 + MAX_ATOMS*N_ELEM
        self.fc1 = nn.Linear(LATENT_DIM + 64, 512)
        self.fc_out = nn.Linear(512, out_dim)
    
    def forward(self, z, sg_idx):
        sg_e = self.sg_emb(sg_idx)
        x = torch.cat([z, sg_e], dim=-1)
        h = F.relu(self.fc1(x))
        out = self.fc_out(h)
        lattice = out[:, :9].view(-1,3,3)
        frac = out[:,9:9+MAX_ATOMS*3].view(-1,MAX_ATOMS,3)
        species_logits = out[:,9+MAX_ATOMS*3:].view(-1,MAX_ATOMS,N_ELEM)
        return lattice, frac, species_logits

class CVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder()
        self.dec = Decoder()
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self, lattice, frac, species_oh, sg_idx):
        mu, logvar = self.enc(lattice, frac, species_oh, sg_idx)
        z = self.reparameterize(mu, logvar)
        lattice_rec, frac_rec, species_logits = self.dec(z, sg_idx)
        return lattice_rec, frac_rec, species_logits, mu, logvar

# Synthetic Dataset for demonstration
class SyntheticCrystalDataset(Dataset):
    """Creates synthetic crystal structure data for training"""
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random but realistic-looking crystal data
        # Lattice matrix (3x3)
        lattice = np.random.uniform(3, 10, (3, 3))
        lattice = lattice @ lattice.T  # Make it symmetric-ish
        
        # Fractional coordinates (MAX_ATOMS x 3)
        frac = np.random.uniform(0, 1, (MAX_ATOMS, 3))
        
        # Species one-hot (MAX_ATOMS x N_ELEM)
        species_oh = np.zeros((MAX_ATOMS, N_ELEM))
        for i in range(MAX_ATOMS):
            species_oh[i, np.random.randint(0, N_ELEM)] = 1.0
        
        # Space group (random between 1-230)
        sg_idx = np.random.randint(1, 231)
        
        return {
            'lattice': torch.FloatTensor(lattice),
            'frac': torch.FloatTensor(frac),
            'species_oh': torch.FloatTensor(species_oh),
            'sg_idx': torch.LongTensor([sg_idx])
        }

def loss_function(lattice_rec, frac_rec, species_logits, 
                  lattice, frac, species_oh, mu, logvar):
    """CVAE loss function"""
    # Reconstruction losses
    lattice_loss = F.mse_loss(lattice_rec, lattice, reduction='sum')
    frac_loss = F.mse_loss(frac_rec, frac, reduction='sum')
    species_loss = F.cross_entropy(
        species_logits.reshape(-1, N_ELEM),
        species_oh.reshape(-1, N_ELEM).argmax(dim=-1),
        reduction='sum'
    )
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    total_loss = lattice_loss + frac_loss + species_loss + 0.1 * kl_loss
    
    return total_loss, {
        'lattice': lattice_loss.item(),
        'frac': frac_loss.item(),
        'species': species_loss.item(),
        'kl': kl_loss.item()
    }

def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    loss_components = {'lattice': 0, 'frac': 0, 'species': 0, 'kl': 0}
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        lattice = batch['lattice'].to(device)
        frac = batch['frac'].to(device)
        species_oh = batch['species_oh'].to(device)
        sg_idx = batch['sg_idx'].squeeze().to(device)
        
        optimizer.zero_grad()
        
        lattice_rec, frac_rec, species_logits, mu, logvar = model(
            lattice, frac, species_oh, sg_idx
        )
        
        loss, components = loss_function(
            lattice_rec, frac_rec, species_logits,
            lattice, frac, species_oh, mu, logvar
        )
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        for key in loss_components:
            loss_components[key] += components[key]
        
        pbar.set_postfix({'loss': f'{loss.item():.2f}'})
    
    avg_loss = total_loss / len(dataloader)
    for key in loss_components:
        loss_components[key] /= len(dataloader)
    
    return avg_loss, loss_components

def main():
    # Configuration
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3
    NUM_SAMPLES = 2000  # Number of synthetic samples
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = CVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Create dataset and dataloader
    print(f"Creating synthetic dataset with {NUM_SAMPLES} samples...")
    dataset = SyntheticCrystalDataset(num_samples=NUM_SAMPLES)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Create checkpoints directory (Windows compatible)
    checkpoint_dir = os.path.join('model', 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoint directory: {os.path.abspath(checkpoint_dir)}")
    
    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    print("=" * 60)
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        avg_loss, components = train_epoch(model, dataloader, optimizer, device)
        
        print(f"Average Loss: {avg_loss:.2f}")
        print(f"  - Lattice: {components['lattice']:.2f}")
        print(f"  - Fractional: {components['frac']:.2f}")
        print(f"  - Species: {components['species']:.2f}")
        print(f"  - KL: {components['kl']:.2f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(checkpoint_dir, 'cvae_best.pt')
            torch.save(model.state_dict(), best_path)
            print(f"✅ Saved best model (loss: {best_loss:.2f})")
        
        # Save latest model every 10 epochs
        if (epoch + 1) % 10 == 0:
            latest_path = os.path.join(checkpoint_dir, 'cvae_latest.pt')
            torch.save(model.state_dict(), latest_path)
            print(f"💾 Saved checkpoint at epoch {epoch+1}")
    
    # Save final model
    final_path = os.path.join(checkpoint_dir, 'cvae_latest.pt')
    torch.save(model.state_dict(), final_path)
    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print(f"📦 Model saved to: {os.path.abspath(final_path)}")
    print(f"🎯 Best loss: {best_loss:.2f}")
    print("=" * 60)
    
    print("\n📋 Next steps:")
    print("1. The model file is ready at: model\\checkpoints\\cvae_latest.pt")
    print("2. Start your Flask server: python app.py")
    print("3. Start your frontend and generate structures!")

if __name__ == "__main__":
    main()