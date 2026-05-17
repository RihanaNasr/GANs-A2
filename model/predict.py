import argparse
import torch
import os
from dataset import DateTokenizer
from rnn_model import RNNModel
from transformer_model import TransformerModel
from vae_model import VAEModel
from gan_model import Generator
#Uses argparse to accept -i (input path), -o (output path), and -m (choice of model).
#Configures execution device (CUDA GPU if available, else CPU).
#Initializes the tokenizer and corresponding model architecture based on -m.
#Loads the trained model weights: model.load_state_dict(torch.load(model_path)).
#Parses conditions from the input file, passes them through the loaded model's generate() loop to get prediction tokens,
#decodes them back to date text strings, and writes them to the specified output file matching the expected format exactly.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to input file")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to output file")
    parser.add_argument("-m", "--model", type=str, default="rnn", choices=["rnn", "transformer", "vae", "gan"], help="Which model to use")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DateTokenizer()
    
    input_vocab_size = len(tokenizer.input_vocab)
    out_vocab_size = len(tokenizer.out_vocab)
    max_len = tokenizer.max_len
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, f"{args.model}_weights.pth")
    
    if args.model == "rnn":
        model = RNNModel(input_vocab_size, out_vocab_size).to(device)
    elif args.model == "transformer":
        model = TransformerModel(input_vocab_size, out_vocab_size).to(device)
    elif args.model == "vae":
        model = VAEModel(input_vocab_size, out_vocab_size, max_len=max_len).to(device)
    elif args.model == "gan":
        model = Generator(input_vocab_size, out_vocab_size, max_len=max_len, noise_dim=64).to(device)
        
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    else:
        print(f"Warning: Model weights {model_path} not found. Generating with untrained model.")
        
    conditions = []
    lines = []
    with open(args.input, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            if len(parts) >= 4:
                day, month, leap, decade = parts[:4]
                try:
                    conditions.append(tokenizer.encode_input(day, month, leap, decade))
                    lines.append(" ".join(parts[:4]))
                except KeyError:
                    pass
                    
    if not conditions:
        print("No valid conditions found in input file.")
        return
        
    conditions_tensor = torch.tensor(conditions, dtype=torch.long).to(device)
    
    batch_size = 128
    all_preds = []
    
    with torch.no_grad():
        for i in range(0, len(conditions_tensor), batch_size):
            batch_c = conditions_tensor[i:i+batch_size]
            if args.model in ["rnn", "transformer"]:
                preds = model.generate(batch_c, max_len, tokenizer.sos_id, tokenizer.eos_id, device)
            elif args.model == "vae":
                preds = model.generate(batch_c)
            elif args.model == "gan":
                noise = torch.randn(batch_c.size(0), 64, device=device)
                preds = model(batch_c, noise, temperature=1.0) # idx returned in eval mode
            all_preds.append(preds.cpu())
            
    all_preds = torch.cat(all_preds, dim=0)
    
    with open(args.output, 'w') as f:
        for i in range(len(lines)):
            date_str = tokenizer.decode_output(all_preds[i])
            f.write(f"{lines[i]} {date_str}\n")
            
if __name__ == "__main__":
    main()
