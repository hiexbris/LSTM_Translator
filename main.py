from datasets import load_dataset
from dataset_class import TranslationDataset
from model import Encoder, Decoder, Seq2Seq
from testing_dataset import TestingDataset
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def plot_attention(tokens_x, tokens_y, attention_list):


    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    """
    tokens_x: List of source tokens (len = source_len)
    tokens_y: List of target tokens (len = target_len)
    attention_list: List of [1, source_len] tensors (len = target_len)
    """

    assert len(tokens_y) == len(attention_list), "Mismatch in target token and attention vector count"
    assert all(attn.shape[1] == len(tokens_x) for attn in attention_list), "Each attention tensor must have correct source length"

    # Convert list of [1, source_len] tensors → [target_len, source_len]
    attention_matrix = torch.cat([attn.squeeze(0).unsqueeze(0) for attn in attention_list], dim=0)
    attention_matrix = attention_matrix.detach().cpu().numpy()  # Shape: [target_len, source_len]

    plt.figure(figsize=(14, 6))
    sns.heatmap(attention_matrix, xticklabels=tokens_x, yticklabels=tokens_y,
                cmap='YlGnBu', linewidths=0.5, cbar=True)

    plt.xlabel("Source Tokens")
    plt.ylabel("Target Tokens")
    plt.title("Attention Heatmap")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def train_model(num_epochs=30, learning_rate=0.001):

    global emb_dim, projected_emb_dim, enc_hidden_dim, dec_hidden_dim, output_dim, attention_size

    dataset = load_dataset("iwslt2017", "iwslt2017-fr-en", trust_remote_code=True)

    train_data = dataset["train"]   
    
    tds = TranslationDataset(train_data,
                            min_freq=3, src_lang='fr', tgt_lang='en',
                            batch_size=64,
                            fr_embedding_path="cc.fr.300.vec.gz",
                            en_embedding_path="cc.en.300.vec.gz",
                            max_size=15000)

    print("French vocab size:", len(tds.src_vocab))
    print("English vocab size:", len(tds.tgt_vocab))

    english_vocab = tds.tgt_vocab
    french_vocab = tds.src_vocab
    vocab_size = len(english_vocab)

    english_emb_matrix = tds.tgt_emb_matrix
    french_emb_matrix = tds.src_emb_matrix

    # Access batched French embeddings and English target indices for training
    x_index_batches = tds.x       # list of tensors, each shape: [batch_size, seq_len]
    y_index_batches = tds.y          # list of tensors, each shape: [batch_size, seq_len]


    torch.save(french_vocab, "french_vocab.pt")
    torch.save(english_vocab, "english_vocab.pt")

    torch.save(french_emb_matrix, "french_emb_matrix.pt")
    torch.save(english_emb_matrix, "english_emb_matrix.pt")

    emb_dim = 300
    projected_emb_dim = 256
    enc_hidden_dim = 512
    dec_hidden_dim = 768
    output_dim = vocab_size
    attention_size = 768

    encoder = Encoder(emb_dim=emb_dim, enc_hidden_dim=enc_hidden_dim)
    decoder = Decoder(output_dim=output_dim, emb_dim=emb_dim, projected_emb_dim=projected_emb_dim, enc_hidden_dim=enc_hidden_dim, dec_hidden_dim=dec_hidden_dim, attention_size=attention_size)
    model = Seq2Seq(encoder=encoder, decoder=decoder, device=device, dec_hidden_dim=dec_hidden_dim, enc_hidden_dim=enc_hidden_dim)
    model = model.to(device)

    checkpoint = torch.load('checkpoints\seq2seq_epoch_20.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    pad_idx = 0 # pad index is 0    

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx) 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    import os
    save_dir = 'checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    best_loss = float('inf')

    from torch.cuda.amp import GradScaler, autocast
    scaler = GradScaler()

    for epoch in range(19, num_epochs):
        model.train()
        total_loss = 0
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        for i, (x_batch, y_batch) in enumerate(zip(x_index_batches, y_index_batches)):

            x_embedded = tds.get_french_embedding_for_indices(x_batch, device)
            y_embedded = tds.get_english_embedding_for_indices(y_batch, device)

            optimizer.zero_grad()

            with autocast():
                outputs = model(x_embedded, y_embedded, tds, mode='train', epoch=epoch+1)  

                outputs = outputs[:, 1:].reshape(-1, vocab_size) # Shift targets to exclude first token (usually <sos>)
                targets = y_batch[:, 1:].reshape(-1)

                loss = criterion(outputs, targets)
            
            total_loss += loss.detach().cpu().item()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            print(f"  Batch {i+1}/{len(x_index_batches)} - Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(x_index_batches)
        print(f"\nEpoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}\n")

        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(save_dir, f'seq2seq_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(save_dir, 'best_seq2seq_model.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'best_loss': best_loss
            }, best_model_path)
            print(f"Best model updated and saved at epoch {epoch+1}")

    torch.save(model.state_dict(), 'seq2seq_model.pt')
    print("Model saved as 'seq2seq_model.pt'.")



# TESTING

def evaluate_model(device, max_len=50):

    dataset = load_dataset("iwslt2017", "iwslt2017-fr-en", trust_remote_code=True)
    test_data = dataset["test"]
    trds = TestingDataset(test_data, french_vocab, english_vocab, french_emb_matrix, english_emb_matrix, batch_size=64)
    x_test_indices_batches = trds.x
    y_test_indices_batches = trds.y 

    encoder = Encoder(emb_dim=emb_dim, enc_hidden_dim=enc_hidden_dim)
    decoder = Decoder(output_dim=output_dim, emb_dim=emb_dim, projected_emb_dim=projected_emb_dim, enc_hidden_dim=enc_hidden_dim, dec_hidden_dim=dec_hidden_dim, attention_size=attention_size)
    model = Seq2Seq(encoder=encoder, decoder=decoder, device=device, dec_hidden_dim=dec_hidden_dim, enc_hidden_dim=enc_hidden_dim)
    model = model.to(device)

    checkpoint = torch.load('checkpoints\seq2seq_epoch_5.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # set to eval mode
    total_tokens = 0
    correct_tokens = 0
    number = 0
    
    with torch.no_grad():
        for x_batch, y_batch in zip(x_test_indices_batches, y_test_indices_batches):

            src_emb = trds.get_french_embedding_for_indices(x_batch, device)  
            
            outputs, predicted_indices = model(src_emb, pre_data=trds, mode='inference', max_len=max_len)
            y_batch = y_batch[:, :max_len]
            
            # Create mask to ignore padding tokens in targets (assuming pad index is 0)
            mask = (y_batch != 0)
            
            correct = (predicted_indices[:, :y_batch.size(1)] == y_batch) & mask
            
            correct_tokens += correct.sum().item()
            total_tokens += mask.sum().item()
            number += 1
            print(f"batch:{number}, accuracy:{100*(correct.sum().item())/mask.sum().item()}")
            if number == 400:
                break
    
    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    print(f'Token-level accuracy on test set: {accuracy * 100:.2f}%')


train_model()

french_vocab = torch.load("french_vocab.pt")
english_vocab = torch.load("english_vocab.pt")
vocab_size = len(english_vocab)
french_emb_matrix = torch.load("french_emb_matrix.pt")
english_emb_matrix = torch.load("english_emb_matrix.pt")

emb_dim = 300
projected_emb_dim = 384
enc_hidden_dim = 512
dec_hidden_dim = 512
output_dim = vocab_size
attention_size = 512
print('Values loaded')

# evaluate_model(device)

def manual_testing():

    from testing_single_data import ManualTestingDataset

    french_sentences = [
        "C'est un voiture."
    ]
    
    for french_sentence in french_sentences:
        raw_data = [french_sentence.lower()]  # List of strings

        test_dataset = ManualTestingDataset(
        raw_data=raw_data,
        src_vocab=french_vocab,
        tgt_vocab=english_vocab,
        src_emb_matrix=french_emb_matrix,
        tgt_emb_matrix=english_emb_matrix,
        batch_size=1
        )

        x_batch = test_dataset.x[0]
        x_embeddings = test_dataset.get_french_embedding_for_indices(x_batch, device)

        encoder = Encoder(emb_dim=emb_dim, enc_hidden_dim=enc_hidden_dim)
        decoder = Decoder(output_dim=output_dim, emb_dim=emb_dim, projected_emb_dim=projected_emb_dim, enc_hidden_dim=enc_hidden_dim, dec_hidden_dim=dec_hidden_dim, attention_size=attention_size)
        model = Seq2Seq(encoder=encoder, decoder=decoder, device=device, dec_hidden_dim=dec_hidden_dim, enc_hidden_dim=enc_hidden_dim)
        model = model.to(device)

        checkpoint = torch.load('checkpoints\seq2seq_epoch_15.pt', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()  

        outputs, predicted_indices, attention = model(x_embeddings, pre_data=test_dataset, mode='inference', max_len=50)
        # predicted_seq_tensor = torch.tensor([predicted_indices], device=model.device)
        translation = test_dataset.decode_indices_to_words(predicted_indices)

        translation = translation[0][5:]
        import re
        tokens_x = re.findall(r"\w+|[^\w\s]", french_sentence, re.UNICODE)
        tokens_x.insert(0, '<SOS>')
        tokens_x.append('<EOS>')

        tokens_y = re.findall(r"\w+|[^\w\s]", translation, re.UNICODE)
        # tokens_y.insert(0, '<SOS>')
        tokens_y.append('<EOS>')
        # print(tokens_x, tokens_y)
        
        plot_attention(tokens_x, tokens_y, attention)


# manual_testing()

