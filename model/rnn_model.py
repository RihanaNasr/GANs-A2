import torch
import torch.nn as nn
import torch.nn.functional as F

#An LSTM model that generates dates step-by-step.
class RNNModel(nn.Module):
    def __init__(self, vocab_size, out_vocab_size, embed_dim=64, hidden_dim=128):
        super().__init__()
        #Dense embedding layer for input conditions
        self.cond_embed = nn.Embedding(vocab_size, embed_dim)
        # Dense embedding layer for output date characters
        self.char_embed = nn.Embedding(out_vocab_size, embed_dim)
        # LSTM expects concatenated embeddings
        self.rnn = nn.LSTM(embed_dim * 2, hidden_dim, batch_first=True)
        # Output head predicting vocab scores
        self.fc = nn.Linear(hidden_dim, out_vocab_size)
    
    #Embeds condition indices: c_emb = self.cond_embed(conditions) (size: [batch, 4, embed_dim])
    #Averages the conditions: c_emb = c_emb.mean(dim=1) to create a single condition vector (size: [batch, embed_dim]).
    #Duplicates condition context for all sequence steps: c_emb = c_emb.unsqueeze(1).repeat(1, seq_len, 1).
    #Concatenates character and condition embeddings: rnn_in = torch.cat([x_emb, c_emb], dim=-1) (size: [batch, seq_len, embed_dim * 2]).
    #Passes it through the LSTM and maps to vocabulary sizes using the linear head (self.fc) to output logits (scores).

    def forward(self, conditions, x):
        # conditions: (batch_size, 4)
        # x: (batch_size, seq_len)
        b, s = x.size()
        
        # embed conditions and average/flatten them
        c_emb = self.cond_embed(conditions) # (b, 4, embed_dim)
        c_emb = c_emb.mean(dim=1) # (b, embed_dim)
        c_emb = c_emb.unsqueeze(1).repeat(1, s, 1) # (b, seq_len, embed_dim)
        
        # embed chars
        x_emb = self.char_embed(x) # (b, seq_len, embed_dim)
        
        rnn_in = torch.cat([x_emb, c_emb], dim=-1) # (b, seq_len, embed_dim*2)
        out, _ = self.rnn(rnn_in)
        logits = self.fc(out) # (b, seq_len, out_vocab_size)
        return logits
     
    # Iterative autoregressive generator
    #1-Initializes a batch of sequence with the <sos> character token.
    #2-Loops up to max_len times
    #3-In each step, feeds current character + condition to LSTM, runs a greedy selection argmax(dim=-1)
    #on the output logits to pick the highest scoring character, appends it, and uses it as the next step input.
#how ana h loop 3shan nrkez 3la l errors(final enhance yarab)
    #Initializes a batch of sequence with the <sos> character token
    #In a loop for a fixed maximum length:
    #Embeds the previous character (or <sos> at start)
    #Concatenates it with the repeated condition embedding
    #Passes through the LSTM to get hidden states and LSTM outputs
    #Uses the output head to predict the next character vocabulary logits
    #Performs greedy decoding (argmax) to select the next character index
    #Appends this index to the generated sequence
    #Feeds it back as input for the next step
    def generate(self, conditions, max_len, sos_id, eos_id, device):
        # conditions: (batch_size, 4)
        b = conditions.size(0)
        c_emb = self.cond_embed(conditions).mean(dim=1) # (b, embed_dim)
        
        hidden = None
        current_char = torch.full((b, 1), sos_id, dtype=torch.long, device=device)
        
        generated = []
        for _ in range(max_len):
            x_emb = self.char_embed(current_char) # (b, 1, embed_dim)
            rnn_in = torch.cat([x_emb, c_emb.unsqueeze(1)], dim=-1)
            
            out, hidden = self.rnn(rnn_in, hidden)
            logits = self.fc(out.squeeze(1)) # (b, out_vocab_size)
            
            # Greedy decoding
            next_char = logits.argmax(dim=-1, keepdim=True) # (b, 1)
            generated.append(next_char)
            current_char = next_char
            
        return torch.cat(generated, dim=1) # (b, max_len)
