import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, emb_dim, enc_hidden_dim, num_layers=1):
        super().__init__()
        self.rnn = nn.LSTM(emb_dim, enc_hidden_dim, num_layers=2, batch_first=True, bidirectional=True)

    def forward(self, src_emb):
        outputs, (hidden, cell) = self.rnn(src_emb)  # outputs: (batch, src_len, 2*hidden)  
        return outputs, hidden, cell


class BahdanauAttention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim, attention_size):
        super().__init__()
        self.W1 = nn.Linear(2*enc_hidden_dim, attention_size)
        self.W2 = nn.Linear(dec_hidden_dim, attention_size)
        self.V = nn.Linear(attention_size, 1)
        self.norm = nn.LayerNorm(attention_size)

    def forward(self, decoder_hidden, encoder_outputs):
        """
        decoder_hidden: (batch_size, dec_hidden_dim)
        encoder_outputs: (batch_size, src_len, enc_hidden_dim)
        """
        decoder_hidden = decoder_hidden.unsqueeze(1)
        # Calculate alignment scores (energy): (batch_size, src_len, dec_hidden_dim)
        energy = torch.tanh(self.W1(encoder_outputs) + self.W2(decoder_hidden))
        energy = self.norm(energy)
        attention_scores = self.V(energy).squeeze(2)  # (batch_size, src_len)
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, src_len)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)  # (batch_size, 1, enc_hidden_dim)
        
        return context, attention_weights


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, projected_emb_dim, enc_hidden_dim, dec_hidden_dim, attention_size):
        super().__init__()
        self.output_dim = output_dim
        self.attention = BahdanauAttention(enc_hidden_dim, dec_hidden_dim, attention_size)
        self.project_emb = nn.Linear(emb_dim, projected_emb_dim)
        self.rnn = nn.LSTM(2*enc_hidden_dim + projected_emb_dim, dec_hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(dec_hidden_dim, output_dim)

    def forward(self, input_token_emb, hidden, cell, encoder_outputs):
        input_token_emb = self.project_emb(input_token_emb)
        context, attn_weights = self.attention(hidden, encoder_outputs)  # (B, 1, E)
        rnn_input = torch.cat((input_token_emb, context), dim=2)  # (B, 1, E+emb)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden.unsqueeze(0), cell.unsqueeze(0)))  # (B, 1, H)
        output = output.squeeze(1)  # (B, H)
        logits = self.fc_out(output)  # (B, V), # no softmax as the loss computation does softmax to outputs and then compute loss
        return logits, hidden.squeeze(0), cell.squeeze(0), attn_weights





class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, dec_hidden_dim, enc_hidden_dim):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.dec_hidden_dim = dec_hidden_dim

        self.enc_hid_to_dec_hid = nn.Linear(2*enc_hidden_dim, dec_hidden_dim)
        self.enc_cell_to_dec_cell = nn.Linear(2*enc_hidden_dim, dec_hidden_dim)

    def beam_search_decode(self, src_emb, pre_data, max_len, beam_width=3):
        batch_size = src_emb.size(0)
        assert batch_size == 1, "Beam search currently only supports batch size of 1"

        # Encode
        encoder_outputs, enc_hidden, enc_cell = self.encoder(src_emb)

        hidden_forward = enc_hidden[-2]
        hidden_backward = enc_hidden[-1]
        combined_hidden = torch.cat([hidden_forward, hidden_backward], dim=1)
        hidden = self.enc_hid_to_dec_hid(combined_hidden)

        cell_forward = enc_cell[-2]
        cell_backward = enc_cell[-1]
        combined_cell = torch.cat([cell_forward, cell_backward], dim=1)
        cell = self.enc_cell_to_dec_cell(combined_cell)

        sos_idx = 2
        eos_idx = 3

        # Initialize beam with <sos>
        beams = [{
            'tokens': [sos_idx],
            'log_prob': 0.0,
            'hidden': hidden,
            'cell': cell
        }]

        completed_beams = []

        for _ in range(max_len):
            new_beams = []
            for beam in beams:
                if beam['tokens'][-1] == eos_idx:
                    completed_beams.append(beam)
                    continue

                input_idx = torch.tensor([[beam['tokens'][-1]]], device=self.device)
                input_emb = pre_data.get_french_embedding_for_indices(input_idx, self.device)

                output, new_hidden, new_cell, _ = self.decoder(
                    input_emb, beam['hidden'], beam['cell'], encoder_outputs
                )

                log_probs = torch.log_softmax(output, dim=1)  # (1, vocab_size)
                top_log_probs, top_indices = log_probs.topk(beam_width)

                for i in range(beam_width):
                    token = top_indices[0][i].item()
                    log_prob = top_log_probs[0][i].item()
                    new_beams.append({
                        'tokens': beam['tokens'] + [token],
                        'log_prob': beam['log_prob'] + log_prob,
                        'hidden': new_hidden,
                        'cell': new_cell
                    })

        # Keep top `beam_width` beams
        beams = sorted(new_beams, key=lambda x: x['log_prob'], reverse=True)[:beam_width]

        completed_beams.extend(beams)
        best_beam = max(completed_beams, key=lambda x: x['log_prob'])

        predicted_indices = torch.tensor(best_beam['tokens'], device=self.device).unsqueeze(0)
        return predicted_indices




    def forward(self, src_emb, tgt_emb=None, pre_data=None, mode='train', max_len=50, epoch=None):
        if mode == 'train':
            batch_size, tgt_len, _ = tgt_emb.shape
            outputs = torch.zeros(batch_size, tgt_len, self.decoder.output_dim).to(self.device)

            encoder_outputs,  enc_hidden, enc_cell = self.encoder(src_emb)

            hidden_forward = enc_hidden[-2]
            hidden_backward = enc_hidden[-1]
            combined_hidden = torch.cat([hidden_forward, hidden_backward], dim=1)
            hidden = self.enc_hid_to_dec_hid(combined_hidden)

            cell_forward = enc_cell[-2]
            cell_backward = enc_cell[-1]
            combined_cell = torch.cat([cell_forward, cell_backward], dim=1)
            cell = self.enc_cell_to_dec_cell(combined_cell)

            # Initialize decoder hidden and cell states as zeros
            # hidden = torch.zeros(batch_size, self.dec_hidden_dim).to(self.device)
            # cell = torch.zeros(batch_size, self.dec_hidden_dim).to(self.device)

            input_token = tgt_emb[:, 0].unsqueeze(1)  # <sos> token embedding

            epoch -= 1
            teacher_forcing_ratio = max(0.5 * (0.95 ** epoch), 0.1)  # decay every epoch

            for t in range(1, tgt_len):
                output, hidden, cell, _ = self.decoder(input_token, hidden, cell, encoder_outputs)
                outputs[:, t] = output  # storing logits/probabilities
                import random
                
                use_teacher_forcing = random.random() < teacher_forcing_ratio

                if use_teacher_forcing:
                    input_token = tgt_emb[:, t].unsqueeze(1)
                else:
                    # probs = torch.softmax(output, dim=1) 
                    pred_token_idx = output.argmax(dim=1)  
                    input_token = pre_data.get_english_embedding_for_indices(pred_token_idx.unsqueeze(1), self.device)

            return outputs
        elif mode == 'inference':

            # predicted_indices = self.beam_search_decode(src_emb, pre_data, max_len, beam_width=3)
            # return None, predicted_indices, None


            ## USE THIS IF WE WANT NO BEAM SEARCH AND ATTENTION WIEGHTS


            batch_size = src_emb.size(0)
            outputs = torch.zeros(batch_size, max_len, self.decoder.output_dim).to(self.device)
            predicted_indices = torch.zeros(batch_size, max_len, dtype=torch.long).to(self.device)
            predicted_indices[:, 0] = 2

            encoder_outputs,  enc_hidden, enc_cell = self.encoder(src_emb)

            hidden_forward = enc_hidden[-2]
            hidden_backward = enc_hidden[-1]
            combined_hidden = torch.cat([hidden_forward, hidden_backward], dim=1)
            hidden = self.enc_hid_to_dec_hid(combined_hidden)

            cell_forward = enc_cell[-2]
            cell_backward = enc_cell[-1]
            combined_cell = torch.cat([cell_forward, cell_backward], dim=1)
            cell = self.enc_cell_to_dec_cell(combined_cell)

            # Initialize decoder hidden and cell states
            # hidden = torch.zeros(batch_size, self.dec_hidden_dim).to(self.device)
            # cell = torch.zeros(batch_size, self.dec_hidden_dim).to(self.device)

            # Get the <sos> token index from x_input
            sos_indices = torch.full((batch_size, 1), 2, dtype=torch.long).to(self.device)
            input_token = pre_data.get_english_embedding_for_indices(sos_indices, self.device) # shape: (B, 1, E)
            attention = []
            

            for t in range(1, max_len):
                output, hidden, cell, attention_wieghts = self.decoder(input_token, hidden, cell, encoder_outputs)  # output: (B, vocab_size)
                outputs[:, t] = output
                
                attention.append(attention_wieghts)

                probs = torch.softmax(output, dim=1) 
                pred_token_idx = probs.argmax(dim=1)  
                predicted_indices[:, t] = pred_token_idx

                if (pred_token_idx == 3).all():  # Assuming 3 is <eos>
                    break

                # Use predicted index for next timestep
                input_token = pre_data.get_english_embedding_for_indices(pred_token_idx.unsqueeze(1), self.device)

            return outputs, predicted_indices, attention

