import torch
import torch.nn as nn
import torch.nn.functional as F

class VoiceConverter(nn.Module):
    def __init__(self, input_dim, target_dim, hidden_dim=768, num_layers=6, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Encoder mit höherer Kapazität
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(input_dim, 512, kernel_size=5, padding=2),  # Von 256 auf 512
            nn.BatchNorm1d(512),
            nn.GELU(),  # ReLU → GELU
            nn.Dropout(dropout),
            nn.Conv1d(512, 1024, kernel_size=3, padding=1),  # Tiefere Schicht
            nn.BatchNorm1d(1024),
            nn.GELU()
        )
        
        # Encoder-Attention hinzugefügt
        self.encoder_attention = nn.MultiheadAttention(2*hidden_dim, num_heads=4, dropout=dropout)
        
        self.encoder_lstm = nn.LSTM(
            input_size=1024,   An Conv-Ausgang angepasst
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout,
            batch_first=True
        )
        
        # VAE Components
        self.fc_mu = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_logvar = nn.Linear(2*hidden_dim, hidden_dim)
        
        # Decoder mit mehr Kapazität
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim + 39,
            hidden_size=2*hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout,
            batch_first=True
        )
        
        self.attention = nn.MultiheadAttention(4*hidden_dim, num_heads=8, dropout=dropout)  # Mehr Heads
        
        self.fc_out = nn.Sequential(
            nn.Linear(8*hidden_dim, 2048),  # Größere Layer
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2048, target_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x, target_stats):
        original_seq_len = x.size(1)
        x = x.permute(0, 2, 1)
        
        # Verbesserter Encoder-Pfad
        conv_out = self.encoder_conv(x)
        conv_out = conv_out.permute(0, 2, 1)
        
        # Encoder-Attention
        attn_in = conv_out.permute(1, 0, 2)
        attn_out, _ = self.encoder_attention(attn_in, attn_in, attn_in)
        conv_out = attn_out.permute(1, 0, 2)
        
        lstm_out, _ = self.encoder_lstm(conv_out)
        stats = torch.mean(lstm_out, dim=1)
        
        mu = self.fc_mu(stats)
        logvar = self.fc_logvar(stats)
        z = self.reparameterize(mu, logvar)
        
        # Decoder
        z = z.unsqueeze(1).expand(-1, original_seq_len, -1)
        target_stats = target_stats.unsqueeze(1).expand(-1, original_seq_len, -1)
        decoder_input = torch.cat([z, target_stats], dim=-1)
        
        lstm_out, _ = self.decoder_lstm(decoder_input)
        
        lstm_out_permuted = lstm_out.permute(1, 0, 2)
        attn_out, _ = self.attention(lstm_out_permuted, lstm_out_permuted, lstm_out_permuted)
        attn_out = attn_out.permute(1, 0, 2)
        
        combined = torch.cat([attn_out, lstm_out], dim=-1)
        return self.fc_out(combined), mu, logvar
