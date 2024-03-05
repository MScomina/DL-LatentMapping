from tasks import trainer
from tasks import latent_mapping

def train_model(dataset_name : str, transformer : bool = False, epochs : int = 200, load_model : bool = False, latent_space : int = 48, dropout : float = 0.5, batch_size : int = 256) -> None:
    if transformer:
        trainer.visual_transformer_model(dim_feedforward=latent_space, nhead=4, num_encoder_layers=3, num_decoder_layers=3, save_name=f"visual_transformer - {latent_space} - {dropout} - (4,3,3)", dataset_name=dataset_name,
                                        batch_size=batch_size, load_model=load_model, epochs=epochs, lr=1e-4, gamma_decay=0.996, epochs_decay=1, dropout=dropout)
    else:
        trainer.autoencoder_model(latent_space_size=latent_space, save_name=f"autoencoder (1) - {latent_space} - {dropout}", dataset_name=dataset_name,
                                  batch_size=batch_size, load_model=load_model, epochs=epochs, lr=1e-4, gamma_decay=0.996, epochs_decay=1, dropout=dropout)
        trainer.autoencoder_model(latent_space_size=latent_space, save_name=f"autoencoder (2) - {latent_space} - {dropout}", dataset_name=dataset_name,
                                    batch_size=batch_size, load_model=load_model, epochs=epochs, lr=1e-4, gamma_decay=0.996, epochs_decay=1, dropout=dropout)
        
def train_dataset(dataset_name : str, transformer : bool = False) -> None:
    print(f"Training {dataset_name} with transformer = {transformer}")
    train_model(dataset_name, load_model=True, latent_space=24, dropout=0.5)
    train_model(dataset_name, load_model=True, latent_space=48, dropout=0.5)
    train_model(dataset_name, load_model=True, latent_space=72, dropout=0.5)
    train_model(dataset_name, load_model=False, latent_space=96, dropout=0.5)
    train_model(dataset_name, load_model=False, latent_space=120, dropout=0.5)
    train_model(dataset_name, load_model=False, latent_space=180, dropout=0.5)
    if transformer:
        train_model(dataset_name, transformer=True, load_model=True, latent_space=48, dropout=0.5)
        
def main():     
    train_dataset("fmnist", transformer=False)
    train_dataset("mnist", transformer=False)
    train_dataset("kmnist", transformer=False)
    train_dataset("cifar10", transformer=True)
    train_dataset("svhn", transformer=True)
    latent_mapping.pca_comparison(n_of_anchors=30, n_of_points=10000, alpha=0.6, k_neighbours=10,
                                  dataset_name="kmnist", use_transformer=True, latent_space_size=48, dim_feedforward=180)


if __name__ == "__main__": 
    main()