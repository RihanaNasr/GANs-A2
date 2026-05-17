import torch
from torch.utils.data import Dataset
import os
# a data Processing & Custom Tokenization
#This file parses raw text data and converts strings into numerical representations (Tensors)
#that PyTorch can compute

#my An encoder-decoder that translates tokens "words/characters"into numerical indices and vice-versa
class DateTokenizer:
    def __init__(self):
        # 1. Define input categories
        self.days = ['[MON]', '[TUE]', '[WED]', '[THU]', '[FRI]', '[SAT]', '[SUN]']
        self.months = ['[JAN]', '[FEB]', '[MAR]', '[APR]', '[MAY]', '[JUN]', '[JUL]', '[AUG]', '[SEP]', '[OCT]', '[NOV]', '[DEC]']
        self.leaps = ['[False]', '[True]']
        self.decades = [f'[{d}]' for d in range(180, 221)]
        ## From [180] (1800s) to [220] (2200s)
        
        # 2. Combine inputs to form vocabulary
        self.input_vocab = self.days + self.months + self.leaps + self.decades
        # Map string -> number ID
        self.input_token2id = {token: idx for idx, token in enumerate(self.input_vocab)}
        # Map number ID -> string
        self.input_id2token = {idx: token for token, idx in self.input_token2id.items()}

        # 3. Define output characters (Date format characters)
        self.out_chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-']
        # Special tokens for sequence control
        self.out_vocab = ['<pad>', '<sos>', '<eos>'] + self.out_chars
        # Map output char -> ID
        self.out_token2id = {token: idx for idx, token in enumerate(self.out_vocab)}
        # Map ID -> output char
        self.out_id2token = {idx: token for token, idx in self.out_token2id.items()}

        # 4. Extract explicit IDs of control tokens
        # Used to align batches to the same length
        self.pad_id = self.out_token2id['<pad>']
        # Start Of Sequence: signals decoder to start
        self.sos_id = self.out_token2id['<sos>']
        # End Of Sequence: signals decoder to stop
        self.eos_id = self.out_token2id['<eos>']
        # Fixed target length ("<sos> 12-12-1980 <eos>")
        self.max_len = 12 # enough for <sos> dd-mm-yyyy <eos>
        
    #Takes string conditions and returns a list of 4 integer IDs mapped from input_token2id
    #Example : encode_input('[MON]', '[JAN]', '[False]', '[198]')
    #Returns : [0, 11, 52, 0]
    def encode_input(self, day, month, leap, decade):
        return [
            self.input_token2id[day],
            self.input_token2id[month],
            self.input_token2id[leap],
            self.input_token2id[decade]
        ]
        
    #Takes a date string (e.g., "12-12-1980"), wraps it with <sos> and <eos> tokens, converts characters to IDs,
    #and pads the remaining space with <pad> tokens to ensure a fixed sequence length of 12.
    #Example : encode_output("12-12-1980")
    #Returns : [1, 29, 51, 23, 43, 45, 51, 45, 49, 54, 56, 2]
    def encode_output(self, date_str):
        ids = [self.sos_id] + [self.out_token2id[c] for c in date_str] + [self.eos_id]
        if len(ids) < self.max_len:
            ids += [self.pad_id] * (self.max_len - len(ids))
        return ids[:self.max_len]
    
    #Takes predicted ID tensors, stops decoding if it encounters eos_id,
    #ignores pad_id and sos_id, and merges remaining characters back to a string (e.g., "12-12-1980").
    #Example : decode_output([1, 29, 51, 23, 43, 45, 51, 45, 49, 54, 56, 2])
    #Returns : "12-12-1980"
    def decode_output(self, ids):
        chars = []
        for i in ids:
            if i == self.eos_id:
                break
            if i not in (self.pad_id, self.sos_id):
                chars.append(self.out_id2token[int(i)])
        return "".join(chars)
#this class for to Wraps the raw dataset into PyTorch dataset format.
class DateDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        self.tokenizer = tokenizer
        self.inputs = []
        self.outputs = []
        # Open data.txt and parse line by line
        with open(data_path, 'r') as f:
            for line in f:
                # remove any leading/trailing whitespace and split by space
                parts = line.strip().split(' ')
                if len(parts) == 5:
                    day, month, leap, decade, date_str = parts
                    # Encode both inputs (conditions) and targets (dates)
                    self.inputs.append(self.tokenizer.encode_input(day, month, leap, decade))
                    self.outputs.append(self.tokenizer.encode_output(date_str))
                    
        #Convert lists to PyTorch LongTensors (integers)
        self.inputs = torch.tensor(self.inputs, dtype=torch.long)
        self.outputs = torch.tensor(self.outputs, dtype=torch.long)
        
    #Returns the total number of lines in the dataset.
    def __len__(self):
        return len(self.inputs)
    
    #Returns a single numerical pair of (input_conditions, target_date) at the requested index
    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]
