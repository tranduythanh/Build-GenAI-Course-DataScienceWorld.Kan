import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from transformer import TranslationModel


# Xử lý dữ liệu NER
class NERDataProcessor:
    def __init__(self):
        # Khởi tạo các danh sách để lưu dữ liệu
        self.sentences = []
        self.words = []
        self.tags = []
        self.current_sentence = []
        self.current_tags = []

    def read_data(self, text_data):
        """
        Đọc dữ liệu từ chuỗi văn bản
        """
        lines = text_data.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line or line.startswith("Sentence:"):
                # Thêm câu hiện tại vào danh sách nếu có
                if self.current_sentence:
                    self.sentences.append(self.current_sentence.copy())
                    self.tags.append(self.current_tags.copy())
                    self.current_sentence = []
                    self.current_tags = []
            else:
                # Tách dữ liệu từng dòng thành các cột
                parts = line.split()
                if len(parts) >= 3:  # Phải có ít nhất 3 cột (Word, POS, Tag)
                    word = parts[0]
                    tag = parts[2]  # Lấy cột Tag cho NER
                    self.current_sentence.append(word)
                    self.current_tags.append(tag)
                    self.words.append(word)

        # Thêm câu cuối cùng nếu còn
        if self.current_sentence:
            self.sentences.append(self.current_sentence.copy())
            self.tags.append(self.current_tags.copy())

        # Tạo ánh xạ từ nhãn sang số và ngược lại
        self.unique_tags = sorted(list(set(tag for tags in self.tags for tag in tags)))
        self.tag2idx = {tag: idx for idx, tag in enumerate(self.unique_tags)}
        self.idx2tag = {idx: tag for tag, idx in self.tag2idx.items()}

        # Tạo danh sách các từ duy nhất
        self.unique_words = sorted(list(set(self.words)))
        self.word2idx = {word: idx+1 for idx, word in enumerate(self.unique_words)}  # +1 để chừa chỗ cho padding
        self.word2idx['<PAD>'] = 0
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

        print(f"Đã đọc {len(self.sentences)} câu với {len(self.unique_words)} từ duy nhất và {len(self.unique_tags)} nhãn duy nhất")
        return self.sentences, self.tags, self.tag2idx, self.idx2tag, self.word2idx, self.idx2word

    def split_data(self, test_size=0.2, random_state=42):
        """
        Chia dữ liệu thành tập train và test
        """
        if len(self.sentences) <= 1:
            # Nếu chỉ có 1 mẫu, sử dụng nó cho cả train và test
            return self.sentences, self.sentences, self.tags, self.tags
        return train_test_split(self.sentences, self.tags, test_size=test_size, random_state=random_state)

# Dataset cho NER
class NERDataset(Dataset):
    def __init__(self, sentences, tags, word2idx, tag2idx, max_len=128):
        self.sentences = sentences
        self.tags = tags
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        # Lấy câu và nhãn tương ứng
        words = self.sentences[idx]
        tags = self.tags[idx]

        # Mã hóa từ và nhãn thành chỉ số
        x = [self.word2idx.get(word, self.word2idx.get('<UNK>', 0)) for word in words]
        y = [self.tag2idx[tag] for tag in tags]

        # Lưu độ dài thực của câu trước khi padding
        seq_len = len(x)

        # Xử lý padding
        if len(x) > self.max_len:
            x = x[:self.max_len]
            y = y[:self.max_len]
            seq_len = self.max_len
        else:
            padding = [0] * (self.max_len - len(x))
            x.extend(padding)
            y.extend([self.tag2idx.get('O', 0)] * (self.max_len - len(y)))

        # Tạo mask để đánh dấu vị trí có giá trị thực
        mask = [1] * seq_len
        if len(mask) < self.max_len:
            mask.extend([0] * (self.max_len - len(mask)))
        mask = mask[:self.max_len]

        # Chuyển sang tensor
        x_tensor = torch.tensor(x, dtype=torch.long)
        y_tensor = torch.tensor(y, dtype=torch.long)
        mask_tensor = torch.tensor(mask, dtype=torch.bool)

        return {
            'words': x_tensor,
            'tags': y_tensor,
            'mask': mask_tensor,
            'seq_len': seq_len  # Thêm thông tin về độ dài thực của câu
        }

# Mô hình BERT cho NER[/]
class BERTNER(nn.Module):
    def __init__(self, vocab_size, num_tags, d_model=256, num_heads=8, d_ff=512, num_layers=4):
        super(BERTNER, self).__init__()

        # Sử dụng mô hình BERT đã thiết kế
        self.bert_encoder = TranslationModel(
            src_vocab_size=vocab_size,
            tgt_vocab_size=num_tags,  # Không quan trọng vì chúng ta chỉ sử dụng encoder
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            enc_layers=num_layers,
            dec_layers=0  # Không sử dụng decoder cho bài toán NER
        ).encoder

        # Lớp phân loại
        self.classifier = nn.Linear(d_model, num_tags)

        # Dropout
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        # Đảm bảo x là tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.long)

        # Đảm bảo mask là tensor nếu có
        if mask is not None and not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.bool)

        # Encoder
        encoded = self.bert_encoder(x, mask)

        # Phân loại
        output = self.dropout(encoded)
        output = self.classifier(output)

        return output

# Hàm huấn luyện mô hình
def train_model(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=10):
    model.train()
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        total_loss = 0

        for batch in train_loader:
            # Đưa dữ liệu lên thiết bị
            words = batch['words'].to(device)
            tags = batch['tags'].to(device)
            mask = batch['mask'].to(device)
            seq_lens = batch.get('seq_len', None)  # Lấy thông tin về độ dài thực nếu có

            # Xóa gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(words, mask)

            # Biến đổi outputs và tags để tính loss
            outputs = outputs.view(-1, outputs.shape[-1])
            tags = tags.view(-1)

            # Chỉ tính loss cho các từ thực (không tính padding)
            active_mask = mask.view(-1)
            active_outputs = outputs[active_mask]
            active_tags = tags[active_mask]

            # Tính loss
            loss = criterion(active_outputs, active_tags)

            # Backward pass và update
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Tính trung bình loss
        avg_train_loss = total_loss / len(train_loader)

        # Đánh giá trên tập validation
        val_loss, val_acc, val_f1 = evaluate_model(model, val_loader, criterion, device)

        # In kết quả
        print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}')

        # Lưu mô hình nếu val loss tốt hơn
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_ner_model.pt')
            print(f'Đã lưu mô hình tốt nhất với Val Loss: {best_val_loss:.4f}')

# Hàm đánh giá mô hình
def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0

    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        for batch in data_loader:
            # Đưa dữ liệu lên thiết bị
            words = batch['words'].to(device)
            tags = batch['tags'].to(device)
            mask = batch['mask'].to(device)
            seq_lens = batch.get('seq_len', None)  # Lấy thông tin về độ dài thực nếu có

            # Forward pass
            outputs = model(words, mask)

            # Biến đổi outputs và tags để tính loss
            outputs = outputs.view(-1, outputs.shape[-1])
            tags = tags.view(-1)

            # Chỉ tính loss cho các từ thực (không tính padding)
            active_mask = mask.view(-1)
            active_outputs = outputs[active_mask]
            active_tags = tags[active_mask]

            # Tính loss
            loss = criterion(active_outputs, active_tags)
            total_loss += loss.item()

            # Lấy predictions
            _, predictions = torch.max(active_outputs, dim=1)

            # Thêm predictions và true labels vào danh sách
            all_predictions.extend(predictions.cpu().tolist())
            all_true_labels.extend(active_tags.cpu().tolist())

    # Tính accuracy
    accuracy = sum([1 for p, t in zip(all_predictions, all_true_labels) if p == t]) / len(all_predictions)

    # Tính F1-score
    tp = sum([1 for p, t in zip(all_predictions, all_true_labels) if p == t and t != 0])  # True positives (không tính O tag)
    fp = sum([1 for p, t in zip(all_predictions, all_true_labels) if p != t and p != 0])  # False positives
    fn = sum([1 for p, t in zip(all_predictions, all_true_labels) if p != t and t != 0])  # False negatives

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    model.train()
    return total_loss / len(data_loader), accuracy, f1

# Hàm đọc dữ liệu từ file CSV
def read_csv_data(file_path):
    df = pd.read_csv(file_path)
    text_data = ""
    current_sentence = ""
    for _, row in df.iterrows():
        if pd.isna(row['Word']):
            # Bắt đầu câu mới
            text_data += current_sentence + "\n"
            current_sentence = ""
        else:
            # Thêm từ vào câu hiện tại
            word = row['Word']
            pos = row['POS']
            tag = row['Tag']
            current_sentence += f"{word} {pos} {tag}\n"

    # Thêm câu cuối cùng nếu có
    if current_sentence:
        text_data += current_sentence

    return text_data

# Chương trình chính
def main(file_path):
    # Đọc dữ liệu từ file CSV
    data_text = read_csv_data(file_path)

    # Xử lý dữ liệu
    processor = NERDataProcessor()
    sentences, tags, tag2idx, idx2tag, word2idx, idx2word = processor.read_data(data_text)
    train_sentences, test_sentences, train_tags, test_tags = processor.split_data(test_size=0.2)

    # Thông tin về dữ liệu
    print(f"Số nhãn duy nhất: {len(tag2idx)}")
    print(f"Các nhãn: {tag2idx}")
    print(f"Số câu trong tập train: {len(train_sentences)}")
    print(f"Số câu trong tập test: {len(test_sentences)}")

    # Tạo datasets
    train_dataset = NERDataset(train_sentences, train_tags, word2idx, tag2idx)
    test_dataset = NERDataset(test_sentences, test_tags, word2idx, tag2idx)

    # Tạo data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Thiết lập thiết bị
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Khởi tạo mô hình
    model = BERTNER(
        vocab_size=len(word2idx),
        num_tags=len(tag2idx),
        d_model=256,
        num_heads=8,
        d_ff=512,
        num_layers=4
    ).to(device)

    # Thiết lập optimizer và loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Huấn luyện mô hình
    train_model(model, train_loader, test_loader, optimizer, criterion, device, num_epochs=30)

    # Đánh giá mô hình trên tập test
    test_loss, test_acc, test_f1 = evaluate_model(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}')

    return model, processor, idx2tag

# Dự đoán nhãn cho một câu mới
def predict(model, sentence, processor, idx2tag, device):
    model.eval()

    # Nếu đang sử dụng MPS (M1/M2 GPU), chuyển mô hình sang CPU để tránh lỗi
    if device.type == 'mps':
        prediction_device = torch.device('cpu')
        model = model.to(prediction_device)
    else:
        prediction_device = device

    # Mã hóa câu
    tokens = [processor.word2idx.get(word, processor.word2idx.get('<UNK>', 0)) for word in sentence]

    # Tạo mask
    mask = [1] * len(tokens)

    # Padding
    if len(tokens) < 128:
        tokens.extend([0] * (128 - len(tokens)))
        mask.extend([0] * (128 - len(mask)))
    else:
        tokens = tokens[:128]
        mask = mask[:128]

    # Chuyển sang tensor và đưa lên thiết bị
    tokens_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    mask_tensor = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)

    # Đưa lên thiết bị
    tokens_tensor = tokens_tensor.to(prediction_device)
    mask_tensor = mask_tensor.to(prediction_device)

    # Dự đoán
    with torch.no_grad():
        outputs = model(tokens_tensor, mask_tensor)
        _, predictions = torch.max(outputs, dim=2)

    # Chuyển kết quả về nhãn
    predicted_tags = [idx2tag[idx.item()] for idx in predictions[0][:len(sentence)]]

    # Chuyển mô hình trở lại thiết bị ban đầu nếu cần
    if device.type == 'mps':
        model = model.to(device)

    return list(zip(sentence, predicted_tags))

# Ví dụ sử dụng
if __name__ == "__main__":
    # Sử dụng file CSV
    csv_file = "ner_adjusted.csv"

    # Kiểm tra và sử dụng GPU nếu có (CUDA hoặc MPS cho Apple Silicon)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')  # Sử dụng GPU của Apple Silicon (M1/M2)
    else:
        device = torch.device('cpu')
    print(f"Sử dụng thiết bị: {device}")

    # Huấn luyện và đánh giá mô hình
    model, processor, idx2tag = main(csv_file)

    # Dự đoán mẫu
    test_sentence = ["John", "visited", "New", "York", "last", "week"]

    # Đảm bảo mô hình được đưa về chế độ đánh giá
    model.eval()

    # Dự đoán
    predictions = predict(model, test_sentence, processor, idx2tag, device)

    print("\nDự đoán mẫu:")
    for word, tag in predictions:
        print(f"{word}: {tag}")