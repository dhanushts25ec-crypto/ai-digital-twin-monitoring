import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# =====================================================================
# 1. GENERATIVE DATA AUGMENTATION (VAE)
# Addresses data scarcity by learning latent space distributions
# =====================================================================
class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super(VariationalAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

def vae_loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# =====================================================================
# 2. GENERATIVE DIGITAL TWIN FOR FAULT DIAGNOSIS & EXPLAINABILITY
# Combines real-time detection, latent sampling, and XAI
# =====================================================================
class GenAIDigitalTwin:
    def __init__(self, input_dim: int, latent_dim: int = 8, anomaly_threshold: float = 0.05):
        self.model = VariationalAutoencoder(input_dim, latent_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.threshold = anomaly_threshold
        self.input_dim = input_dim

    def train_baseline(self, normal_data: torch.Tensor, epochs: int = 50):
        """Trains VAE on nominal operational sensor streams."""
        self.model.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            recon, mu, logvar = self.model(normal_data)
            loss = vae_loss_function(recon, normal_data, mu, logvar)
            loss.backward()
            self.optimizer.step()

    def generate_synthetic_faults(self, num_samples: int) -> torch.Tensor:
        """Generates synthetic rare failure scenarios to augment training data."""
        self.model.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.model.fc_mu.out_features)
            synthetic_data = self.model.decoder(z)
        return synthetic_data

    def diagnose_and_explain(self, sensor_stream: torch.Tensor):
        """
        Calculates anomaly score (reconstruction error) and extracts feature-level 
        attributions for human-in-the-loop decision support (XAI).
        """
        self.model.eval()
        with torch.no_grad():
            recon, _, _ = self.model(sensor_stream)
            
            # Reconstruction error per feature
            feature_errors = torch.abs(sensor_stream - recon)
            total_anomaly_score = torch.mean(feature_errors, dim=-1).item()
            
            is_anomaly = total_anomaly_score > self.threshold
            
            # Simple XAI attribution: rank features by contribution to error
            feature_contributions = feature_errors.squeeze().numpy()
            explanation = {
                f"sensor_channel_{i}": float(contrib) 
                for i, contrib in enumerate(feature_contributions)
            }
            
            return {
                "anomaly_detected": is_anomaly,
                "anomaly_score": total_anomaly_score,
                "feature_attribution_xai": explanation
            }

# =====================================================================
# 3. FEDERATED LEARNING INTERFACE
# Enables multi-twin collaboration without sharing raw data
# =====================================================================
def get_model_parameters(twin: GenAIDigitalTwin):
    """Extracts weights for secure cross-site model aggregation."""
    return twin.model.state_dict()

def set_model_parameters(twin: GenAIDigitalTwin, state_dict: dict):
    """Updates local Digital Twin with globally aggregated parameters."""
    twin.model.load_state_dict(state_dict)

# =====================================================================
# DEMONSTRATION OF EXECUTION
# =====================================================================
if __name__ == "__main__":
    # Simulate 10 sensor features (e.g., vibration, thermal, pressure)
    num_sensors = 10
    nominal_data = torch.randn(100, num_sensors)
    
    # Initialize and train Digital Twin
    dt_system = GenAIDigitalTwin(input_dim=num_sensors)
    dt_system.train_baseline(nominal_data, epochs=10)
    
    # 1. Augmentation: Generate rare fault samples
    synthetic_faults = dt_system.generate_synthetic_faults(num_samples=5)
    print(f"Generated {synthetic_faults.shape[0]} synthetic fault profiles.")
    
    # 2. Real-time Diagnosis & XAI on incoming test sample
    test_sample = torch.randn(1, num_sensors) + 2.5  # Simulate elevated readings
    diagnosis = dt_system.diagnose_and_explain(test_sample)
    
    print("\n--- Diagnostic Report ---")
    print(f"Anomaly Status: {diagnosis['anomaly_detected']}")
    print(f"Anomaly Score:  {diagnosis['anomaly_score']:.4f}")
    print("XAI Feature Error Attribution Top Channel:")
    sorted_xai = sorted(diagnosis['feature_attribution_xai'].items(), key=lambda x: x[1], reverse=True)
    print(f"  Highest anomaly contributor: {sorted_xai[0]}")
