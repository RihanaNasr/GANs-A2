import torch
import torch.nn as nn
import torch.nn.functional as F
#Conditional Generative Adversarial Network
#Generator creates fake dates conditioned on input categories.
#Discriminator tries to distinguish real dates from fake ones, also conditioned on categories.
#Gumbel-Softmax used for discrete sampling in the generator.

#Contains two networks training in a minimax game: Generator vs. Discriminator.
class Generator(nn.Module):
    def __init__(self, vocab_size, out_vocab_size, max_len=12, noise_dim=64, embed_dim=64, hidden_dim=256):
        super().__init__()
        self.max_len = max_len #Input Date Maximum length
        self.out_vocab_size = out_vocab_size #Size of vocabulary
        self.cond_embed = nn.Embedding(vocab_size, embed_dim)#Embedding layer for condition inputs
        # MLP layers to project (Noise Vector + Condition Embeddings) -> Flattened sequence logit scores
        self.fc = nn.Sequential(
            nn.Linear(noise_dim + embed_dim * 4, hidden_dim),#Concatenates noise (64) and condition embeddings (4×64=256) -> 320 inputs
            nn.BatchNorm1d(hidden_dim),#Normalizes activations across the batch
            nn.LeakyReLU(0.2),#Applies non-linearity
            nn.Linear(hidden_dim, hidden_dim * 2),#Expands capacity
            nn.BatchNorm1d(hidden_dim * 2),#Normalizes activations across the batch
            nn.LeakyReLU(0.2),#Applies non-linearity
            nn.Linear(hidden_dim * 2, max_len * out_vocab_size)#Generates output logits of shape (batch, max_len * out_vocab_size)
        )
    #Forward Pass: Inputs are category embeddings and a random noise vector.
    #It concatenates them and passes them through the MLP to produce logits for each token position.
    #Gumbel-Softmax Sampling: During training, it uses Gumbel-Softmax to generate fake sequences in a way that allows gradients to flow back to the noise input.
    def forward(self, conditions, noise, temperature=1.0):
        b = conditions.size(0)#Batch size
        c_emb = self.cond_embed(conditions).view(b, -1) #(b, 4 * embed_dim)
        
        gen_in = torch.cat([noise, c_emb], dim=1)#Concatenates noise (64) and condition embeddings (4×64=256) -> 320 inputs
        out = self.fc(gen_in)#Passes through MLP to produce logits for each token position
        out = out.view(b, self.max_len, self.out_vocab_size)#Reshapes to (batch, max_len, out_vocab_size)
        
        # Use Gumbel-Softmax for differentiable discrete sampling
        #In training, standard argmax is non-differentiable (stops backprop).
        #Gumbel-Softmax performs a differentiable continuous approximation of discrete choices, allowing gradients to flow from the Discriminator back into the Generator.
        if self.training:
            out = F.gumbel_softmax(out, tau=temperature, hard=True)
        else:
            #During inference (eval), it applies standard greedy argmax decoding to output the final character indices.
            # During inference, we just want the hard predictions
            idx = out.argmax(dim=-1)
            out = F.one_hot(idx, num_classes=self.out_vocab_size).float()
            return idx # Return indices for generation
            
        return out
#Discriminator
#Architecture: A multi-layer perceptron (MLP) that flattens the input date sequence and concatenates it with the condition embeddings.
#Purpose: It acts as a binary classifier, outputting a single logit score indicating the probability of the input being real.
class Discriminator(nn.Module):
    def __init__(self, vocab_size, out_vocab_size, max_len=12, embed_dim=64, hidden_dim=256):
        super().__init__()
        self.cond_embed = nn.Embedding(vocab_size, embed_dim)#Embedding layer for condition inputs
        
        self.fc = nn.Sequential(
            nn.Linear(max_len * out_vocab_size + embed_dim * 4, hidden_dim * 2),#Concatenates flattened date (12×80=960) and condition embeddings (4×64=256) -> 1216 inputs
            nn.LeakyReLU(0.2),#Applies non-linearity
            nn.Linear(hidden_dim * 2, hidden_dim),#Expands capacity
            nn.LeakyReLU(0.2),#Applies non-linearity
            nn.Linear(hidden_dim, 1)#Outputs a single scalar logit
        )
    #Forward Pass: Takes the condition embeddings and the one-hot encoded date sequence as input.
    #It flattens the date sequence, concatenates it with the condition embeddings, and passes the result through the MLP to produce a single classification score.
    def forward(self, conditions, x_onehot):
        # x_onehot: (b, max_len, out_vocab_size)
        b = conditions.size(0)#Batch size
        c_emb = self.cond_embed(conditions).view(b, -1)#(b, 4 * embed_dim)
        
        x_flat = x_onehot.view(b, -1)#Flattens the one-hot encoded date sequence
        disc_in = torch.cat([x_flat, c_emb], dim=1)#Concatenates flattened date with condition embeddings
        
        return self.fc(disc_in)
#Flattens the one-hot date sequence representation, concatenates it with the condition embeddings, and outputs a single value per sequence.
#Higher logit represents "real date," lower logit represents "fake date generated by G."