from datasets import load_dataset

# Tải tập dữ liệu
raw_datasets = load_dataset("conll2003")

# In thông tin về tập dữ liệu
print("Thông tin về tập dữ liệu:")
print(f"Số lượng mẫu train: {len(raw_datasets['train'])}")
print(f"Số lượng mẫu validation: {len(raw_datasets['validation'])}")
print(f"Số lượng mẫu test: {len(raw_datasets['test'])}")
print("\nCác features của tập dữ liệu:")
print(raw_datasets['train'].features)
print("\nNhãn NER có trong tập dữ liệu:")
print(raw_datasets['train'].features['ner_tags'].feature.names)

# In 10 mẫu đầu tiên của tập train theo định dạng gốc
print("10 mẫu đầu tiên của tập train theo định dạng gốc:")
for i in range(10):
    print(f"\nMẫu {i+1}:")
    # Lấy các thông tin
    tokens = raw_datasets['train'][i]['tokens']
    pos_tags = raw_datasets['train'][i]['pos_tags']
    chunk_tags = raw_datasets['train'][i]['chunk_tags']
    ner_tags = raw_datasets['train'][i]['ner_tags']
    
    # In từng dòng theo định dạng gốc
    for token, pos, chunk, ner in zip(tokens, pos_tags, chunk_tags, ner_tags):
        # Chuyển đổi số thành nhãn
        pos_label = raw_datasets['train'].features['pos_tags'].feature.names[pos]
        chunk_label = raw_datasets['train'].features['chunk_tags'].feature.names[chunk]
        ner_label = raw_datasets['train'].features['ner_tags'].feature.names[ner]
        
        # In theo định dạng gốc
        print(f"{token:<15} {pos_label:<8} {chunk_label:<8} {ner_label}")
    print("-" * 50) 