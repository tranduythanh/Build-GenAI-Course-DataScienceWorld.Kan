import torch
import json
from transformer import TranslationModel

def main():
    print("Starting debug script...")
    
    # Check device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Try to load the SQuAD data
    try:
        print("Loading SQuAD data...")
        with open('dev-v2.0.json', 'r', encoding='utf-8') as f:
            squad_data = json.load(f)
        print(f"SQuAD data loaded successfully. Version: {squad_data.get('version', 'unknown')}")
        print(f"Number of articles: {len(squad_data.get('data', []))}")
    except Exception as e:
        print(f"Error loading SQuAD data: {e}")
    
    # Try to initialize the TranslationModel
    try:
        print("Initializing TranslationModel...")
        model = TranslationModel(
            src_vocab_size=1000,
            tgt_vocab_size=2,
            d_model=256,
            num_heads=8,
            d_ff=512,
            enc_layers=4,
            dec_layers=0
        )
        print("TranslationModel initialized successfully")
        print(f"Model structure: {model}")
    except Exception as e:
        print(f"Error initializing TranslationModel: {e}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"Error in main: {e}")
        traceback.print_exc()
