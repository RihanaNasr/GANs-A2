import torch
import torch.nn as nn
import math
#Uses parallel Self-Attention layers but enforces a causal mask so it cannot look at future tokens.
#this Injects sinusoidal positional information directly into character embeddings so the Transformer knows the order/position of digits.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

#Transformer Generator Architecture
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, out_vocab_size, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.cond_embed = nn.Embedding(vocab_size, d_model)# Condition Embedder
        self.char_embed = nn.Embedding(out_vocab_size, d_model)# Character Embedder
        self.pos_encoder = PositionalEncoding(d_model)# Positional Encoder
        # Multi-head Self Attention layers
        # Single TransformerEncoderLayer with desired parameters
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        # Wrapper to stack multiple layers
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Linear layer to map transformer output to vocabulary size
         # Output Head
        self.fc = nn.Linear(d_model, out_vocab_size)
#Generates a triangular attention mask.
#Values are 0 for past tokens (attention allowed) and -inf for future tokens (attention blocked).
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
#Forward Pass: Embeds conditions and digits, concatenates them, applies the causal mask, and passes through the Transformer.
#1- Concatenates condition embeddings as a prefix (prompt) to the character embeddings: src = torch.cat([c_emb, x_emb], dim=1).
#2- Applies the causal mask specifically to the character part: mask[4:, 4:] = seq_mask (first 4 elements represent conditions, which are always fully visible).
#3- Feeds the combined sequence to the Transformer.
#4- Passes the transformer output through the linear layer.
#5- Returns the output for the sequence part (excluding the condition tokens).
    def forward(self, conditions, x):
        # conditions: (b, 4)
        # x: (b, seq_len)
        b, s = x.size()
        
        c_emb = self.cond_embed(conditions) # (b, 4, d_model)
        x_emb = self.char_embed(x) # (b, s, d_model)
        x_emb = self.pos_encoder(x_emb)
        
        # Concat conditions as prefix to the sequence
        # src: (b, 4 + s, d_model)
        src = torch.cat([c_emb, x_emb], dim=1)
        
        # We need a mask to prevent attending to future characters in x
        # Conditions can attend to each other, x characters can attend to conditions and past characters
        # sz = 4 + s
        sz = src.size(1)
        mask = torch.zeros(sz, sz, device=x.device)
        # Sequence part cannot attend to future sequence parts
        seq_mask = self.generate_square_subsequent_mask(s).to(x.device)
        mask[4:, 4:] = seq_mask
        
        out = self.transformer(src, mask=mask) # (b, 4+s, d_model)
        
        # Extract the sequence part
        out_seq = out[:, 4:, :] # (b, s, d_model)
        return self.fc(out_seq)
        
    #Autoregressive generation (one token at a time) with masking.
    #Starts with <sos>, computes causal mask, feeds through the network,
    #gets the prediction score of the last token using logits.argmax,
    #appends it to the sequence, and loops.
    def generate(self, conditions, max_len, sos_id, eos_id, device):
        b = conditions.size(0)
        c_emb = self.cond_embed(conditions) # (b, 4, d_model)
        
        current_seq = torch.full((b, 1), sos_id, dtype=torch.long, device=device)
        
        for _ in range(max_len):
            s = current_seq.size(1)
            x_emb = self.char_embed(current_seq)
            x_emb = self.pos_encoder(x_emb)
            
            src = torch.cat([c_emb, x_emb], dim=1)
            sz = src.size(1)
            mask = torch.zeros(sz, sz, device=device)
            mask[4:, 4:] = self.generate_square_subsequent_mask(s).to(device)
            
            out = self.transformer(src, mask=mask)
            logits = self.fc(out[:, -1, :]) # (b, out_vocab_size)
            
            next_char = logits.argmax(dim=-1, keepdim=True)
            current_seq = torch.cat([current_seq, next_char], dim=1)
            
        return current_seq[:, 1:] # exclude initial sos_id