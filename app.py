import pygame
import torch
import sys
import re
import time
from pygame import gfxdraw
from testing_single_data import ManualTestingDataset
from model import Encoder, Decoder, Seq2Seq
# --- Translation Functions ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load vocabularies and embeddings
french_vocab = torch.load("french_vocab.pt")
english_vocab = torch.load("english_vocab.pt")
french_emb_matrix = torch.load("french_emb_matrix.pt")
english_emb_matrix = torch.load("english_emb_matrix.pt")
# Model configuration
emb_dim = 300
enc_hidden_dim = 512
dec_hidden_dim = 768
attention_size = 768
# Initialize model
encoder = Encoder(emb_dim=emb_dim, enc_hidden_dim=enc_hidden_dim)
decoder = Decoder(output_dim=len(english_vocab), emb_dim=emb_dim,
                projected_emb_dim=256, enc_hidden_dim=enc_hidden_dim,
                dec_hidden_dim=dec_hidden_dim, attention_size=attention_size)
model = Seq2Seq(encoder, decoder, device, dec_hidden_dim, enc_hidden_dim)
model.load_state_dict(torch.load('checkpoints/seq2seq_epoch_15.pt', map_location=device)['model_state_dict'])
model.to(device)
model.eval()
def plot_attention(input, output, attention_list):
    import matplotlib.pyplot as plt
    import seaborn as sns
    tokens_x = re.findall(r"\w+|[^\w\s]", input, re.UNICODE)
    tokens_x.insert(0, '<SOS>')
    tokens_x.append('<EOS>')
    tokens_y = re.findall(r"\w+|[^\w\s]", output, re.UNICODE)
    # tokens_y.insert(0, '<SOS>')
    tokens_y.append('<EOS>')
    assert len(tokens_y) == len(attention_list), "Mismatch in target token and attention vector count"
    assert all(attn.shape[1] == len(tokens_x) for attn in attention_list), "Each attention tensor must have correct source length"
    # Convert list of [1, source_len] tensors → [target_len, source_len]
    attention_matrix = torch.cat([attn.squeeze(0).unsqueeze(0) for attn in attention_list], dim=0)
    attention_matrix = attention_matrix.detach().cpu().numpy()  # Shape: [target_len, source_len]
    plt.figure(figsize=(10, 6))
    sns.heatmap(attention_matrix, xticklabels=tokens_x, yticklabels=tokens_y,
                cmap='YlGnBu', linewidths=0.5, cbar=True)
    plt.xlabel("Source Tokens")
    plt.ylabel("Target Tokens")
    plt.title("Attention Heatmap")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
def translate_sentence(sentence):
    try:
        dataset = ManualTestingDataset(
            raw_data=[sentence.lower()],
            src_vocab=french_vocab,
            tgt_vocab=english_vocab,
            src_emb_matrix=french_emb_matrix,
            tgt_emb_matrix=english_emb_matrix,
            batch_size=1
        )
        x_batch = dataset.x[0]
        x_embeddings = dataset.get_french_embedding_for_indices(x_batch, device)
        outputs, predicted_indices, attention = model(x_embeddings, pre_data=dataset, 
                                                    mode='inference', max_len=50)
        translation = dataset.decode_indices_to_words(predicted_indices)[0][5:]
        return translation, dataset, attention
    except Exception as e:
        print(f"Translation error: {e}")
        return "Translation Error", None, None
# --- UI Configuration ---
WINDOW_SIZE = (1280, 720)
COLORS = {
    'bg': (18, 18, 29),
    'primary': (94, 234, 212),
    'secondary': (58, 82, 140),
    'text': (240, 240, 240),
    'error': (255, 89, 94),
    'history': (30, 30, 46)
}
class AnimatedValue:
    def __init__(self, value, speed=0.2):
        self.target = value
        self.current = value
        self.speed = speed
    def update(self, dt):
        self.current += (self.target - self.current) * self.speed * dt * 60
        return self.current
class TextInput:
    def __init__(self, rect, font_size=28):
        self.rect = pygame.Rect(rect)
        self.font = pygame.font.Font(None, font_size)
        self.text = ""
        self.cursor_pos = 0
        self.selection_start = None
        self.selection_end = None
        self.last_blink = time.time()
        self.cursor_visible = True
        self.animation = AnimatedValue(self.rect.y)
        self.scroll_offset = 0
        self.dragging = False   
    
    def handle_mouse(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.dragging = True
                x = event.pos[0] - self.rect.x - 15
                self.cursor_pos = self.pos_from_x(x)
                self.selection_start = self.cursor_pos
                self.selection_end = self.cursor_pos
                
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
            
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            x = event.pos[0] - self.rect.x - 15
            self.selection_end = self.pos_from_x(x)
            self.cursor_pos = self.selection_end
    def pos_from_x(self, x):
        pos = 0
        width = 0
        for i in range(len(self.text)):
            char_width = self.font.size(self.text[i])[0]
            if width + char_width/2 > x:
                return pos
            width += char_width
            pos += 1
        return pos
    def draw(self, surface, is_active):
        current_y = self.animation.update(0.016)
        animated_rect = self.rect.copy()
        animated_rect.y = current_y
        
        # Draw background
        pygame.draw.rect(surface, COLORS['secondary'], animated_rect, border_radius=12)
        pygame.draw.rect(surface, COLORS['primary'] if is_active else COLORS['secondary'], 
                    animated_rect, 2, border_radius=12)
        
        # Draw text with wrapping
        text = self.text[self.scroll_offset:]
        text_width = self.font.size(text)[0]

        visible_width = self.rect.width - 30
        cursor_x = self.font.size(self.text[:self.cursor_pos])[0]
        if cursor_x - self.scroll_offset > visible_width - 20:
            self.scroll_offset = cursor_x - visible_width + 20
        elif cursor_x - self.scroll_offset < 20:
            self.scroll_offset = max(0, cursor_x - 20)
        
        # Draw selection highlight
        if self.selection_start is not None and self.selection_end is not None:
            start = min(self.selection_start, self.selection_end) - self.scroll_offset
            end = max(self.selection_start, self.selection_end) - self.scroll_offset
            if start >= 0:
                x_start = self.font.size(text[:start])[0]
                x_end = self.font.size(text[:end])[0]
                pygame.draw.rect(surface, COLORS['primary'], 
                            (animated_rect.x+15+x_start, animated_rect.y+5, x_end-x_start, animated_rect.height-10))
        
        # Draw text
        text_surf = self.font.render(text, True, COLORS['text'])
        surface.blit(text_surf, (animated_rect.x+15, animated_rect.centery - text_surf.get_height()//2))
        
        # Draw cursor
        if is_active and self.cursor_visible:
            cursor_pos = self.cursor_pos - self.scroll_offset
            if cursor_pos >= 0:
                cursor_x = self.font.size(text[:cursor_pos])[0]
                pygame.draw.line(surface, COLORS['primary'], 
                            (animated_rect.x+15+cursor_x, animated_rect.y+10),
                            (animated_rect.x+15+cursor_x, animated_rect.bottom-10), 2)
class HistoryItem:
    def __init__(self, input_text, output_text):
        self.input_text = input_text
        self.output_text = output_text
        self.animation = AnimatedValue(0)
    def draw(self, surface, rect, font):
        alpha = int(self.animation.current * 255)
        bg = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(bg, (*COLORS['history'], alpha), bg.get_rect(), border_radius=8)
        
        # Wrap text
        padding = 5
        max_width = rect.width - 20
        y = padding
        
        # Draw input text
        input_lines = self.wrap_text(self.input_text, font, max_width)
        for line in input_lines:
            input_surf = font.render(line, True, (255, 255, 255, alpha))
            surface.blit(bg, rect.topleft)
            surface.blit(input_surf, (rect.x+10, rect.y + y))
            y += font.get_height()
        
        # Draw output text
        output_lines = self.wrap_text(self.output_text, font, max_width)
        for line in output_lines:
            output_surf = font.render(line, True, (200, 200, 255, alpha))
            surface.blit(output_surf, (rect.x+10, rect.y + y))
            y += font.get_height()
    
    def wrap_text(self, text, font, max_width):
        lines = []
        words = text.split()
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            width = font.size(test_line)[0]
            if width <= max_width:
                current_line.append(word)
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        return lines
class GradientButton:
    def __init__(self, rect, text):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.hover = AnimatedValue(0)
    def draw(self, surface, mouse_pos):
        self.hover.target = 1 if self.rect.collidepoint(mouse_pos) else 0
        progress = self.hover.update(0.016)
        
        # Background
        bg = pygame.Surface(self.rect.size, pygame.SRCALPHA)
        pygame.draw.rect(bg, COLORS['primary'], bg.get_rect(), border_radius=8)
        bg.set_alpha(int(100 + 155 * progress))
        surface.blit(bg, self.rect.topleft)
        
        # Text
        font = pygame.font.Font(None, 24)
        text_surf = font.render(self.text, True, COLORS['text'])
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)
class TranslationApp:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(WINDOW_SIZE)  # First create window
        pygame.display.set_caption("Neural Translator")
        pygame.scrap.init()  # Then initialize clipboard
        self.clock = pygame.time.Clock()
        
        # UI Elements
        self.history = []
        self.font = pygame.font.Font(None, 22)
        self.loading = False
        self.input = TextInput((50, 80, 800, 60))     # Reduced width
        self.output = TextInput((50, 180, 800, 150))  # Taller output box
        self.translate_btn = GradientButton((50, 350, 200, 50), "Translate")
        self.attention_btn = GradientButton((270, 350, 200, 50), "Show Attention")
        self.history_panel_rect = pygame.Rect(900, 60, 350, 600)  # Moved left
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION):
                self.input.handle_mouse(event)
            if event.type == pygame.MOUSEBUTTONDOWN:
                if self.translate_btn.rect.collidepoint(event.pos):
                    self.trigger_translation()
                elif self.attention_btn.rect.collidepoint(event.pos):
                    self.show_attention()
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE:
                    if pygame.key.get_mods() & pygame.KMOD_CTRL:
                        # Delete entire word
                        text_before = self.input.text[:self.input.cursor_pos]
                        last_space = text_before.rfind(' ')
                        if last_space == -1:
                            new_pos = 0
                        else:
                            new_pos = last_space + 1
                        self.input.text = self.input.text[:new_pos] + self.input.text[self.input.cursor_pos:]
                        self.input.cursor_pos = new_pos
                    else:
                        # Normal backspace
                        if len(self.input.text) > 0:
                            self.input.text = self.input.text[:self.input.cursor_pos-1] + self.input.text[self.input.cursor_pos:]
                            self.input.cursor_pos = max(0, self.input.cursor_pos - 1)
                elif event.key == pygame.K_DELETE:
                    if self.input.cursor_pos < len(self.input.text):
                        self.input.text = self.input.text[:self.input.cursor_pos] + self.input.text[self.input.cursor_pos+1:]
                elif event.key == pygame.K_LEFT:
                    self.input.cursor_pos = max(0, self.input.cursor_pos - 1)
                elif event.key == pygame.K_RIGHT:
                    self.input.cursor_pos = min(len(self.input.text), self.input.cursor_pos + 1)
                elif event.key == pygame.K_RETURN:  # Handle Enter key
                    self.trigger_translation()
                elif event.unicode.isprintable():
                    self.input.text = (self.input.text[:self.input.cursor_pos] + 
                                    event.unicode + 
                                    self.input.text[self.input.cursor_pos:])
                    self.input.cursor_pos += 1
    def draw_loading(self):
        t = time.time()
        for i in range(8):
            angle = t * 5 + i * 45
            # Create vector using CLASS METHOD (not instance method)
            vec = pygame.math.Vector2.from_polar((20, angle))  # (radius, angle)
            pos = (
                self.translate_btn.rect.centerx + vec.x,
                self.translate_btn.rect.centery + vec.y
            )
            pygame.draw.circle(self.screen, COLORS['primary'], pos, 3)
    def trigger_translation(self):
        """Handle translation with proper UI feedback"""
        self.loading = True
        self.draw_ui()  # Force UI update to show loading
        translation, dataset, attention = translate_sentence(self.input.text)
        self.output.text = translation
        self.current_attention = (self.input.text, translation, attention)
        self.history.append(HistoryItem(self.input.text[:30], translation[:50]))
        self.loading = False
    
    def draw_output_text(self):
        """Custom text wrapping for output box"""
        output_rect = self.output.rect.copy()
        output_rect.y = self.output.animation.update(0.016)
        
        # Draw box
        pygame.draw.rect(self.screen, COLORS['secondary'], output_rect, border_radius=12)
        pygame.draw.rect(self.screen, COLORS['secondary'], output_rect, 2, border_radius=12)
        
        # Text wrapping
        font = self.output.font
        text = self.output.text
        words = text.split(' ')
        space_width = font.size(' ')[0]
        max_width = output_rect.width - 30
        x, y = output_rect.x + 15, output_rect.y + 10
        line_height = font.get_height()
        
        current_line = []
        for word in words:
            test_line = current_line + [word]
            test_width = font.size(' '.join(test_line))[0]
            
            if test_width <= max_width:
                current_line.append(word)
            else:
                # Render current line
                line_surf = font.render(' '.join(current_line), True, COLORS['text'])
                self.screen.blit(line_surf, (x, y))
                y += line_height
                current_line = [word]
        
        # Render remaining text
        if current_line:
            line_surf = font.render(' '.join(current_line), True, COLORS['text'])
            self.screen.blit(line_surf, (x, y))
    def draw_ui(self):
        """Separate UI drawing for forced updates"""
        self.screen.fill(COLORS['bg'])
        
        # Input and output boxes
        self.input.draw(self.screen, True)
        self.draw_output_text()  # Custom output drawing
        
        # Buttons and history
        self.translate_btn.draw(self.screen, pygame.mouse.get_pos())
        self.attention_btn.draw(self.screen, pygame.mouse.get_pos())
        self.draw_history()
        
        if self.loading:
            self.draw_loading()
        
        # Background elements
        t = time.time()
        for i in range(15):
            x = (t * 50 + i*100) % (WINDOW_SIZE[0]+200) - 100
            y = WINDOW_SIZE[1] * 0.8 + 50*(i%3)
            pygame.draw.circle(self.screen, (*COLORS['primary'], 10 + i*2), (x, y), 20 + 10*(i%3))
        
        pygame.display.flip()
    
    def draw_history(self):
        panel = pygame.Surface((300, 600), pygame.SRCALPHA)
        pygame.draw.rect(panel, (*COLORS['history'], 200), (0, 0, 300, 600), border_radius=12)
        
        y = 20
        for item in reversed(self.history[-5:]):
            item.draw(panel, pygame.Rect(20, y, 260, 60), self.font)
            y += 70
        
        self.screen.blit(panel, (950, 60))
    def run(self):
        while True:
            self.handle_events()
            self.draw_ui()
            self.clock.tick(60)
    def show_attention(self):
        """Show attention plot with proper error handling"""
        if hasattr(self, 'current_attention'):
            input_text, output_text, attention = self.current_attention
            if attention:
                plot_attention(input_text, output_text, attention)
            else:
                self.output.text = "No attention data available"
        else:
            self.output.text = "Translate something first!"
if __name__ == "__main__":
    app = TranslationApp()
    app.run()