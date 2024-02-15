from tasks import trainer

def main():
    trainer.visual_transformer_model(dim_feedforward=48, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                                      batch_size=256, load_model=True, epochs=100, lr=1e-4, gamma_decay=0.993, epochs_decay=1, dropout=0.0)
    trainer.autoencoder_model(linear_layers=2,
                              batch_size=128, load_model=True, epochs=200, lr=1e-4, gamma_decay=0.9, epochs_decay=10, dropout=0.0)

if __name__ == "__main__": 
    main()