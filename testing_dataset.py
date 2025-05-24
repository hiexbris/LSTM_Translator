import torch
import numpy as np
from collections import Counter
import re
# MADE ONLY SO VOCAB AND EMBEDDING MATRIX STAYS SAME ALONG TRAINING AND TESTING DATASET, needed as output depends on vocab
class TestingDataset:
    def __init__(self, raw_data, src_vocab, tgt_vocab, src_emb_matrix, tgt_emb_matrix, src_lang='fr', tgt_lang='en', batch_size=32,
                 device=None):
        """
        raw_data: list of dicts like {'translation': {'fr': [...], 'en': [...]}}, or similar
        src_lang, tgt_lang: language codes in the dataset
        batch_size: batch size for padding and batching
        fr_embedding_path, en_embedding_path: paths to FastText gzipped embeddings
        device: torch device to use for tensors (cuda/cpu)
        """
        self.batch_size = batch_size
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        raw_data = raw_data.map(lambda x: {'src_len': len(x['translation'][src_lang].split())})
        raw_data = raw_data.sort('src_len')

        raw_data = raw_data.map(
            lambda x: {
                'translation': {
                    src_lang: x['translation'][src_lang].lower(),
                    tgt_lang: x['translation'][tgt_lang].lower()
                }
            }
        )

        print(raw_data[0])
        
        # Extract raw sentences
        self.src_sentences = [item['translation'][src_lang] for item in raw_data]
        self.tgt_sentences = [item['translation'][tgt_lang] for item in raw_data]
        
        # Tokenize
        self.src_sentences = [self.tokenize(s) for s in self.src_sentences]
        self.tgt_sentences = [self.tokenize(s) for s in self.tgt_sentences]
        
        # Build vocabs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        
        # Load embeddings
        self.src_emb_matrix = src_emb_matrix
        self.tgt_emb_matrix = tgt_emb_matrix
        
        # Prepare batches with padding and SOS/EOS tokens
        self.batches = self.batch_and_pad(self.src_sentences, self.tgt_sentences, batch_size)
        
        # Precompute tgt (Y) batches as indices only (no one-hot)
        self.x_indices, self.y_indices = self.all_batches_to_indices(self.batches, self.src_vocab, self.tgt_vocab, self.device)
    
    @staticmethod
    def tokenize(text):
        return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

    @staticmethod
    def add_sos_eos(sentences, sos_token='<sos>', eos_token='<eos>'):
        return [[sos_token] + sent + [eos_token] for sent in sentences]

    @staticmethod
    def pad_sentence(sentence, max_len, pad_token='<pad>'):
        if len(sentence) > max_len:
            return sentence[:max_len]
        else:
            return sentence + [pad_token] * (max_len - len(sentence))

    def batch_and_pad(self, X, Y, batch_size, pad_token='<pad>'):
        batches = []
        n = len(X)
        for i in range(0, n, batch_size):
            batch_X = X[i:i+batch_size]
            batch_Y = Y[i:i+batch_size]

            batch_X = self.add_sos_eos(batch_X)
            batch_Y = self.add_sos_eos(batch_Y)

            max_len_X = max(len(sent) for sent in batch_X)
            max_len_Y = max(len(sent) for sent in batch_Y)

            if max_len_X <= 50:
                padded_batch_X = [self.pad_sentence(sent, max_len_X, pad_token) for sent in batch_X]
                padded_batch_Y = [self.pad_sentence(sent, max_len_Y, pad_token) for sent in batch_Y]

                batches.append((padded_batch_X, padded_batch_Y))
        return batches

    def all_batches_to_indices(self, batches, src_vocab, tgt_vocab, device):
        x_indices = []
        y_indices = []
        for padded_X_batch, padded_Y_batch in batches:
            batch_size = len(padded_X_batch)
            seq_len_x = len(padded_X_batch[0])
            seq_len_y = len(padded_Y_batch[0])

            x_idx_batch = torch.zeros((batch_size, seq_len_x), dtype=torch.long)
            y_idx_batch = torch.zeros((batch_size, seq_len_y), dtype=torch.long)

            for i, sentence in enumerate(padded_X_batch):
                for j, word in enumerate(sentence):
                    idx = src_vocab.get(word, src_vocab.get('<unk>', 0))
                    x_idx_batch[i, j] = idx

            for i, sentence in enumerate(padded_Y_batch):
                for j, word in enumerate(sentence):
                    idx = tgt_vocab.get(word, tgt_vocab.get('<unk>', 0))
                    y_idx_batch[i, j] = idx

            x_indices.append(x_idx_batch.to(device))
            y_indices.append(y_idx_batch.to(device))

        return x_indices, y_indices
    
    def get_french_embedding_for_indices(self, indices_tensor, device):
        """
        indices_2d: Tensor or list of shape (batch_size, seq_len) with French word indices
        Returns: embedding tensor of shape (batch_size, seq_len, embedding_dim)
        """
        # indices_tensor = torch.tensor(indices_2d, dtype=torch.long, device=device)
        vocab_size = len(self.src_vocab)
        indices_tensor.to(device)
        # Clamp indices to valid range
        indices_tensor = torch.clamp(indices_tensor, 0, vocab_size - 1)
        
        emb_matrix_tensor = torch.tensor(self.src_emb_matrix, dtype=torch.float32, device=device)  # (vocab_size, emb_dim)
        
        embeddings = emb_matrix_tensor[indices_tensor]  # fancy indexing, shape: (batch_size, seq_len, emb_dim)
        return embeddings


    def get_english_onehot_for_indices(self, indices_tensor):
        """
        indices_2d: Tensor or list of shape (batch_size, seq_len) with word indices
        Returns: one-hot tensor of shape (batch_size, seq_len, vocab_size)
        """
        vocab_size = len(self.tgt_vocab)
        # indices_tensor = torch.tensor(indices_2d, dtype=torch.long)  # Ensure tensor
        
        batch_size, seq_len = indices_tensor.shape
        onehot = torch.zeros(batch_size, seq_len, vocab_size, dtype=torch.float32)
        
        # Flatten indices for scatter_
        flat_indices = indices_tensor.view(-1, 1)  # shape (batch_size*seq_len, 1)
        flat_onehot = onehot.view(-1, vocab_size)  # shape (batch_size*seq_len, vocab_size)
        
        flat_onehot.scatter_(1, flat_indices, 1.0)
        
        return onehot
    
    def get_english_embedding_for_indices(self, indices_tensor, device):
        """
        indices_2d: Tensor or list of shape (batch_size, seq_len) with word indices
        Returns: embedding tensor of shape (batch_size, seq_len, embedding_dim)
        """
        # indices_tensor = torch.tensor(indices_2d, dtype=torch.long).to(device)
        vocab_size = len(self.tgt_vocab)
        indices_tensor.to(device)
        # Clamp indices to valid range
        indices_tensor = torch.clamp(indices_tensor, 0, vocab_size - 1)
        
        emb_matrix_tensor = torch.tensor(self.tgt_emb_matrix, dtype=torch.float32, device=device)  # (vocab_size, emb_dim)
        
        embeddings = emb_matrix_tensor[indices_tensor]  # fancy indexing, shape: (batch_size, seq_len, emb_dim)
        return embeddings


    # Properties to access data conveniently
    @property
    def x(self):
        """French embeddings batches (list of tensors, CPU)"""
        return self.x_indices
    @property
    def y(self):
        """English target batches as index tensors (list of tensors, on device)"""
        return self.y_indices
    
    

    

