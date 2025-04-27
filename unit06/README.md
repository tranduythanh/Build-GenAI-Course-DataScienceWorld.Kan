# Nhận dạng Thực thể có Tên (NER) với BERT

Dự án này triển khai hệ thống Nhận dạng Thực thể có Tên (NER) sử dụng mô hình BERT trên tập dữ liệu CoNLL-2003. Triển khai bao gồm toàn bộ quy trình từ xử lý dữ liệu đến huấn luyện và đánh giá mô hình.

## 1. Tập dữ liệu

`CoNLL-2003` là một tập dữ liệu chuẩn cho bài toán NER (tiếng Anh)

### Định dạng dữ liệu thô
Mỗi câu trong tập dữ liệu được biểu diễn dưới dạng một chuỗi các token, mỗi token có 4 thông tin:
1. Từ (word)
2. Nhãn POS (Part-of-Speech)
3. Nhãn chunk
4. Nhãn NER

Ví dụ một câu trong tập dữ liệu:
```
EU      NNP     B-NP    B-ORG
rejects VBZ     B-VP    O
German  JJ      B-NP    B-MISC
call    NN      I-NP    O
to      TO      B-VP    O
boycott VB      I-VP    O
British JJ      B-NP    B-MISC
lamb    NN      I-NP    O
.       .       O       O
```

Giải thích:
- Cột 1: Từ trong câu
- Cột 2: Nhãn POS (ví dụ: NNP = Proper noun, VBZ = Verb, 3rd person singular present)
- Cột 3: Nhãn chunk (ví dụ: B-NP = Beginning of Noun Phrase, I-NP = Inside Noun Phrase)
- Cột 4: Nhãn NER (ví dụ: B-ORG = Beginning of Organization, B-MISC = Beginning of Miscellaneous, O = Outside)

Trong bài toán này, ta chỉ quan tâm đến cột 1 va 4.

Tập dữ liệu được chia thành 3 phần: 
- Tập huấn luyện (Train set): 14,041 câu
- Tập xác thực (Validation set): 3,250 câu
- Tập kiểm thử (Test set): 3,453 câu

Tổng cộng có 20,744 sentences trong toàn bộ tập dữ liệu CoNLL-2003. Đây là một tập dữ liệu khá lớn và cân đối, với tỷ lệ phân chia:
- Train: ~67.7%
- Validation: ~15.7%
- Test: ~16.6%

## 2. Tiền xử lý dữ liệu

### Tách token
Sử dụng `BERT tokenizer` để tách token cho câu, với các tham số:
  - `is_split_into_words=True`: dữ liệu đầu vào đã được tách thành từng từ (word). Ở đây, 1 word sẽ ứng với 1 token.
  - `truncation=True`: cắt bớt câu nếu vượt quá độ dài tối đa
  - `padding=True`: thêm padding để các câu có cùng độ dài

### Chuẩn hoá nhãn (label):

Với mỗi câu, lấy danh sách `word_ids` từ `tokenizer`, duyệt qua từng token và gán nhãn NER tương ứng:
- Token đặc biệt `[CLS]`/`[SEP]`: gán nhãn `-100`. Giá trị -100 là một quy ước đặc biệt trong PyTorch, nó báo cho hàm tính loss biết cần bỏ qua các vị trí này khi tính loss.
- Token đầu tiên của một từ: gán nhãn của từ đó
- Sub-token tiếp theo: gán nhãn của từ gốc hoặc `-100` tùy theo tham số `label_all_tokens`
- Kết quả là một tập dữ liệu mới, trong đó:
  - Các token đã được tách và padding
  - Nhãn đã được căn chỉnh với các token
  - Đã loại bỏ các cột không cần thiết

Ví dụ:
Sau khi tokenize bởi BERT:

```
Tokens:   [CLS] Peter lives  in  New   York  [SEP]

           ↓      ↓     ↓    ↓    ↓     ↓     ↓

word_ids: None    0     1    2    3     3     None

           ↓      ↓     ↓    ↓    ↓     ↓     ↓

Nhãn:     -100  B-PER   O    O  B-LOC  I-LOC -100
```

Đối với câu ngắn hơn (cần padding):

```
Tokens:  [CLS] John   went to Paris [SEP] [PAD] [PAD] [PAD]

          ↓     ↓      ↓    ↓   ↓     ↓     ↓     ↓     ↓

Word IDs: None  0      1    2   3    None  None  None  None

          ↓     ↓      ↓    ↓   ↓     ↓     ↓     ↓     ↓

Nhãn:     -100  B-PER  O    O  B-LOC -100  -100  -100  -100
```

## 2. Khởi tạo mô hình

### Kiến trúc mô hình
- Sử dụng `BERT-base-cased` làm backbone
- Thêm một lớp `classification head` để dự đoán nhãn NER
- Lưu ý, gán `num_labels = len(label_list)`, tức là số lượng nhãn NER trong tập dữ liệu

```python
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id,
)
```

### Data Collator
- DataCollatorForTokenClassification: xử lý padding và căn chỉnh cho batch dữ liệu
- Đảm bảo các chuỗi trong batch có cùng độ dài

## 3. Huấn luyện mô hình

### Tham số huấn luyện
```python
args = TrainingArguments(
    output_dir="ner-bert-conll03",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=50,
)
```

### Thiết lập Trainer
- Sử dụng HuggingFace Trainer để quản lý quá trình huấn luyện
- Cấu hình các thành phần:
  - Mô hình
  - Tham số huấn luyện
  - Tập dữ liệu train và validation
  - Tokenizer
  - Data collator
  - Tính toán metrics

## 4. Đánh giá mô hình

### Các chỉ số đánh giá
- Sử dụng `seqeval metric` để đánh giá hiệu suất NER
- Tính toán các chỉ số:
  - Precision: tỷ lệ dự đoán đúng trong số các dự đoán
  - Recall: tỷ lệ dự đoán đúng trong số các thực thể thật
  - F1-score: trung bình điều hòa của precision và recall

```python
def compute_metrics(p):
    preds, labels = p
    preds = np.argmax(preds, axis=-1)
    true_preds, true_labels = [], []
    for pred, lab in zip(preds, labels):
        for p_i, l_i in zip(pred, lab):
            if l_i != -100:
                true_preds.append(label_list[p_i])
                true_labels.append(label_list[l_i])
    results = metric.compute(predictions=[true_preds], references=[true_labels])
    return {k: results[f"overall_{k}"] for k in ["precision", "recall", "f1"]}
```

## 5. Ap dụng mô hình

### Thiết lập Pipeline
```python
ner = pipeline(
    "token-classification",
    model="ner-bert-conll03",
    tokenizer="ner-bert-conll03",
    aggregation_strategy="simple"
)
```

### Ví dụ
```python
sentence = "Barack Obama visited Hà Nội in 2016."
print(ner(sentence))
```
