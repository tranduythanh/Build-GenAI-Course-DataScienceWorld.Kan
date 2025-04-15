import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
from sklearn.model_selection import train_test_split
from transformer import TranslationModel

# Data processor for SQuAD 2.0
class SQuADProcessor:
    def __init__(self, max_seq_length=384, max_query_length=64, doc_stride=128):
        self.max_seq_length = max_seq_length
        self.max_query_length = max_query_length
        self.doc_stride = doc_stride
        self.vocab = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[UNK]': 3}
        self.vocab_size = 4  # Start with special tokens

    def build_vocab(self, texts):
        """Build vocabulary from texts"""
        for text in texts:
            for word in text.split():
                if word not in self.vocab:
                    self.vocab[word] = self.vocab_size
                    self.vocab_size += 1
        return self.vocab, self.vocab_size

    def tokenize(self, text):
        """Improved tokenization with basic preprocessing"""
        # Convert to lowercase and remove punctuation for better matching
        text = text.lower()
        # Replace common punctuation with spaces
        for char in ',.!?;:()[]{}"':
            text = text.replace(char, ' ')
        # Split on whitespace and filter out empty tokens
        return [token for token in text.split() if token]

    def convert_tokens_to_ids(self, tokens):
        """Convert tokens to ids using vocabulary"""
        return [self.vocab.get(token, self.vocab['[UNK]']) for token in tokens]

    def read_squad_examples(self, file_path):
        """Read SQuAD examples from file"""
        print(f"Reading SQuAD examples from {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            squad_data = json.load(f)

        examples = []
        all_texts = []  # For building vocabulary

        for article in squad_data['data']:
            title = article['title']
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                all_texts.append(context)

                for qa in paragraph['qas']:
                    qas_id = qa['id']
                    question = qa['question']
                    all_texts.append(question)
                    is_impossible = qa.get('is_impossible', False)

                    if not is_impossible:
                        answer = qa['answers'][0]  # Take the first answer
                        answer_text = answer['text']
                        answer_start = answer['answer_start']

                        # Find the token index of the answer in the context
                        context_tokens = self.tokenize(context)
                        char_to_word_offset = {}

                        # Map character positions to token positions
                        offset = 0
                        for i, token in enumerate(context_tokens):
                            for j in range(len(token)):
                                char_to_word_offset[offset + j] = i
                            offset += len(token) + 1  # +1 for the space

                        # Find start and end token indices
                        start_token_idx = char_to_word_offset.get(answer_start, 0)
                        end_token_idx = char_to_word_offset.get(min(answer_start + len(answer_text) - 1,
                                                                   len(context) - 1), 0)
                    else:
                        answer_text = ""
                        start_token_idx = 0
                        end_token_idx = 0

                    example = {
                        'qas_id': qas_id,
                        'title': title,
                        'question': question,
                        'context': context,
                        'answer_text': answer_text,
                        'start_token_idx': start_token_idx,
                        'end_token_idx': end_token_idx,
                        'is_impossible': is_impossible
                    }
                    examples.append(example)

        # Build vocabulary from all texts
        print(f"Building vocabulary from {len(all_texts)} texts...")
        self.build_vocab(all_texts)

        print(f"Loaded {len(examples)} examples from {file_path}")
        print(f"Vocabulary size: {self.vocab_size}")
        return examples

    def convert_examples_to_features(self, examples):
        """Convert examples to features that can be fed to the model"""
        print("Converting examples to features...")
        features = []

        for i, example in enumerate(examples):
            if i % 1000 == 0:
                print(f"Processing example {i}/{len(examples)}")

            question = example['question']
            context = example['context']

            # Tokenize question
            question_tokens = self.tokenize(question)
            if len(question_tokens) > self.max_query_length:
                question_tokens = question_tokens[:self.max_query_length]

            # Tokenize context
            context_tokens = self.tokenize(context)

            # Create spans with overlap
            spans = []
            max_tokens = self.max_seq_length - len(question_tokens) - 3  # [CLS], [SEP], [SEP]

            for span_start in range(0, len(context_tokens), self.doc_stride):
                span_end = min(span_start + max_tokens, len(context_tokens))
                span = context_tokens[span_start:span_end]
                spans.append((span_start, span_end, span))

                if span_end >= len(context_tokens):
                    break

            for span_start, span_end, span in spans:
                # Create tokens: [CLS] question [SEP] context [SEP]
                tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + span + ['[SEP]']

                # Create token type IDs: 0 for question, 1 for context
                token_type_ids = [0] * (len(question_tokens) + 2) + [1] * (len(span) + 1)

                # Convert tokens to IDs
                input_ids = self.convert_tokens_to_ids(tokens)

                # Create attention mask: 1 for real tokens, 0 for padding
                attention_mask = [1] * len(input_ids)

                # Pad to max_seq_length
                padding_length = self.max_seq_length - len(input_ids)
                if padding_length > 0:
                    input_ids += [0] * padding_length
                    attention_mask += [0] * padding_length
                    token_type_ids += [0] * padding_length

                # Truncate if too long
                input_ids = input_ids[:self.max_seq_length]
                attention_mask = attention_mask[:self.max_seq_length]
                token_type_ids = token_type_ids[:self.max_seq_length]

                # Find answer positions in the tokenized context
                start_position = None
                end_position = None

                if not example['is_impossible']:
                    # Adjust for the current span
                    orig_start_idx = example['start_token_idx']
                    orig_end_idx = example['end_token_idx']

                    # Check if the answer is in this span
                    if orig_start_idx >= span_start and orig_end_idx < span_end:
                        # Adjust for [CLS] question [SEP]
                        start_position = orig_start_idx - span_start + len(question_tokens) + 2
                        end_position = orig_end_idx - span_start + len(question_tokens) + 2
                    else:
                        # Answer not in this span
                        # Use 0 as a valid position for the [CLS] token
                        start_position = 0  # [CLS] token
                        end_position = 0
                else:
                    # For impossible questions, set positions to [CLS] token
                    start_position = 0
                    end_position = 0

                feature = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'token_type_ids': token_type_ids,
                    'start_position': start_position,
                    'end_position': end_position,
                    'is_impossible': 1 if example['is_impossible'] else 0,
                    'example_id': example['qas_id']
                }
                features.append(feature)

        print(f"Converted {len(examples)} examples to {len(features)} features")
        return features

    def process_data(self, file_path, max_examples=1000):
        """Process SQuAD data from file with a limit on examples"""
        examples = self.read_squad_examples(file_path)

        # Limit the number of examples for faster processing
        if max_examples > 0 and max_examples < len(examples):
            print(f"Limiting to {max_examples} examples for faster processing")
            examples = examples[:max_examples]

        features = self.convert_examples_to_features(examples)
        return examples, features, self.vocab, self.vocab_size

# Dataset for SQuAD
class SQuADDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]

        return {
            'input_ids': torch.tensor(feature['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(feature['attention_mask'], dtype=torch.long),
            'token_type_ids': torch.tensor(feature['token_type_ids'], dtype=torch.long),
            'start_position': torch.tensor(feature['start_position'], dtype=torch.long),
            'end_position': torch.tensor(feature['end_position'], dtype=torch.long),
            'is_impossible': torch.tensor(feature['is_impossible'], dtype=torch.float),
            'example_id': feature['example_id']
        }

# BERT-based model for SQuAD using the custom transformer
class BERTForQuestionAnswering(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_heads=4, d_ff=256, num_layers=2, dropout_rate=0.1):
        super(BERTForQuestionAnswering, self).__init__()

        # Use the encoder from TranslationModel
        self.bert_encoder = TranslationModel(
            src_vocab_size=vocab_size,
            tgt_vocab_size=2,  # Not important as we only use the encoder
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            enc_layers=num_layers,
            dec_layers=0  # No decoder needed
        ).encoder

        # Output layers for start and end positions
        self.qa_outputs = nn.Linear(d_model, 2)

        # Output layer for answerable classification
        self.qa_classifier = nn.Linear(d_model, 1)

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # Ensure input_ids is a tensor
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long)

        # Ensure attention_mask is a tensor if provided
        if attention_mask is not None and not isinstance(attention_mask, torch.Tensor):
            attention_mask = torch.tensor(attention_mask, dtype=torch.bool)
        else:
            # Create default attention mask if none provided
            attention_mask = (input_ids != 0).long()

        # Use token_type_ids if provided (for segment embeddings)
        # This helps the model distinguish between question and context
        if token_type_ids is not None and not isinstance(token_type_ids, torch.Tensor):
            token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)

        # Get encoder outputs - pass token_type_ids to encoder if it supports it
        # For now, we'll just use input_ids and attention_mask
        sequence_output = self.bert_encoder(input_ids, attention_mask)

        # Apply dropout
        sequence_output = self.dropout(sequence_output)

        # Get logits for start and end positions
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # Get logits for answerable classification (using [CLS] token)
        cls_output = sequence_output[:, 0, :]
        answerable_logits = self.qa_classifier(cls_output).squeeze(-1)

        return start_logits, end_logits, answerable_logits

# Calculate Exact Match and F1 score
def calculate_metrics(predictions, labels):
    exact_match = 0
    f1_sum = 0
    answerable_accuracy = 0

    for example_id in predictions:
        pred = predictions[example_id]
        label = labels[example_id]

        # Track answerable accuracy separately
        if pred['is_impossible'] == label['is_impossible']:
            answerable_accuracy += 1

        # Check if both prediction and label agree on impossibility
        if pred['is_impossible'] == label['is_impossible']:
            if pred['is_impossible'] == 1:
                # Both agree it's impossible, count as exact match
                exact_match += 1
                f1_sum += 1
            else:
                # Both agree it's answerable, check span
                if pred['start_idx'] == label['start_idx'] and pred['end_idx'] == label['end_idx']:
                    exact_match += 1
                    f1_sum += 1
                else:
                    # Calculate F1 based on overlap
                    pred_span = set(range(pred['start_idx'], pred['end_idx'] + 1))
                    label_span = set(range(label['start_idx'], label['end_idx'] + 1))

                    if len(pred_span) == 0 and len(label_span) == 0:
                        f1_sum += 1
                    elif len(pred_span) == 0 or len(label_span) == 0:
                        f1_sum += 0
                    else:
                        intersection = len(pred_span.intersection(label_span))
                        precision = intersection / len(pred_span) if len(pred_span) > 0 else 0
                        recall = intersection / len(label_span) if len(label_span) > 0 else 0

                        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                        f1_sum += f1

    metrics = {
        'exact_match': exact_match / len(predictions),
        'f1': f1_sum / len(predictions),
        'answerable_accuracy': answerable_accuracy / len(predictions)
    }

    return metrics

# Prediction function
def predict(model, processor, question, context, device, max_seq_length=384):
    model.eval()

    # Tokenize input
    question_tokens = processor.tokenize(question)
    context_tokens = processor.tokenize(context)

    # Truncate if needed
    if len(question_tokens) > processor.max_query_length:
        question_tokens = question_tokens[:processor.max_query_length]

    # Create input sequence: [CLS] question [SEP] context [SEP]
    tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + context_tokens

    # Truncate if too long
    if len(tokens) > max_seq_length - 1:  # -1 for the final [SEP]
        tokens = tokens[:max_seq_length - 1]
    tokens.append('[SEP]')

    # Create token type IDs: 0 for question, 1 for context
    token_type_ids = [0] * (len(question_tokens) + 2) + [1] * (len(tokens) - len(question_tokens) - 2)

    # Convert tokens to IDs
    input_ids = processor.convert_tokens_to_ids(tokens)

    # Create attention mask
    attention_mask = [1] * len(input_ids)

    # Pad to max_seq_length
    padding_length = max_seq_length - len(input_ids)
    if padding_length > 0:
        input_ids += [0] * padding_length
        attention_mask += [0] * padding_length
        token_type_ids += [0] * padding_length

    # Truncate if still too long
    input_ids = input_ids[:max_seq_length]
    attention_mask = attention_mask[:max_seq_length]
    token_type_ids = token_type_ids[:max_seq_length]

    # Convert to tensors and move to device
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
    attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(device)
    token_type_ids = torch.tensor([token_type_ids], dtype=torch.long).to(device)

    # Get predictions
    with torch.no_grad():
        start_logits, end_logits, answerable_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

    # Check if the question is answerable - use a higher threshold to avoid false negatives
    # The model seems to be biased toward classifying questions as unanswerable
    is_impossible = torch.sigmoid(answerable_logits[0]) >= 0.8

    if is_impossible:
        return "Unanswerable question"

    # Get the most likely start and end positions
    # Ensure we only consider valid positions (not in question or special tokens)
    # The valid range starts after [SEP] token following the question
    question_end_idx = len(question_tokens) + 1  # +1 for [CLS] and +1 for [SEP]

    # Apply a mask to prevent selecting positions in the question
    start_mask = torch.zeros_like(start_logits)
    start_mask[0, question_end_idx:] = 1
    masked_start_logits = start_logits * start_mask

    # Get the most likely start position
    start_idx = torch.argmax(masked_start_logits, dim=1)[0].item()

    # Only consider end positions after the predicted start position
    end_mask = torch.zeros_like(end_logits)
    end_mask[0, start_idx:] = 1
    masked_end_logits = end_logits * end_mask
    end_idx = torch.argmax(masked_end_logits, dim=1)[0].item()

    # If we got invalid indices, fall back to the original approach
    if start_idx == 0 and torch.sum(start_mask) > 0:
        start_idx = torch.argmax(start_logits, dim=1)[0].item()
        end_idx = torch.argmax(end_logits[0, start_idx:], dim=0).item() + start_idx

    # Extract answer tokens
    answer_tokens = tokens[start_idx:end_idx+1]

    # Convert tokens to text
    answer = ' '.join(answer_tokens)

    # Clean up answer
    answer = answer.replace('[CLS]', '').replace('[SEP]', '').strip()

    # If answer is empty or only contains special tokens, return "No answer found"
    if not answer or answer.isspace():
        return "No answer found"

    return answer

# Main function
def main():
    print("Starting SQuAD 2.0 Question Answering script...")

    # Force CPU usage for initial testing
    device = torch.device('cpu')
    print(f"Using device: {device}")

    # Process data (limit to 1000 examples for faster processing)
    processor = SQuADProcessor()
    _, features, _, vocab_size = processor.process_data('dev-v2.0.json', max_examples=100000)

    # Split data into train and validation sets
    train_features, val_features = train_test_split(features, test_size=0.1, random_state=42)

    # Create datasets
    train_dataset = SQuADDataset(train_features)
    val_dataset = SQuADDataset(val_features)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # Create data loaders with smaller batch size
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)

    # Initialize model
    print(f"Initializing model with vocabulary size: {vocab_size}")
    model = BERTForQuestionAnswering(vocab_size=vocab_size)
    model.to(device)

    # Set number of epochs
    num_epochs = 10
    print("Training model...")

    # Initialize optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)

    # Learning rate scheduler
    total_steps = len(train_loader) * num_epochs
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-5, total_steps=total_steps)
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training
        model.train()
        total_train_loss = 0
        train_steps = 0

        for batch in train_loader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            start_positions = batch['start_position'].to(device)
            end_positions = batch['end_position'].to(device)
            is_impossible = batch['is_impossible'].to(device)

            # Forward pass
            start_logits, end_logits, answerable_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

            # Calculate loss
            # For start and end positions
            # Ensure all positions are valid (non-negative)
            # CrossEntropyLoss requires targets to be non-negative and < num_classes
            valid_start_positions = torch.clamp(start_positions, min=0, max=start_logits.size(1)-1)
            valid_end_positions = torch.clamp(end_positions, min=0, max=end_logits.size(1)-1)

            start_loss = nn.CrossEntropyLoss()(start_logits, valid_start_positions)
            end_loss = nn.CrossEntropyLoss()(end_logits, valid_end_positions)
            span_loss = (start_loss + end_loss) / 2

            # For answerable classification
            # Use a weighted loss to address the bias toward unanswerable questions
            # This gives more weight to answerable examples (when is_impossible=0)
            pos_weight = torch.tensor([0.3]).to(device)  # Lower weight for unanswerable examples
            answerable_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(answerable_logits, is_impossible)

            # Combine losses with less weight on answerable classification
            loss = span_loss + 0.3 * answerable_loss  # Reduce the weight of answerable classification

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            train_steps += 1

        avg_train_loss = total_train_loss / train_steps
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        total_val_loss = 0
        val_steps = 0
        all_predictions = {}
        all_labels = {}

        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                start_positions = batch['start_position'].to(device)
                end_positions = batch['end_position'].to(device)
                is_impossible = batch['is_impossible'].to(device)
                example_ids = batch['example_id']

                # Forward pass
                start_logits, end_logits, answerable_logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )

                # Calculate loss
                # Ensure all positions are valid (non-negative)
                valid_start_positions = torch.clamp(start_positions, min=0, max=start_logits.size(1)-1)
                valid_end_positions = torch.clamp(end_positions, min=0, max=end_logits.size(1)-1)

                start_loss = nn.CrossEntropyLoss()(start_logits, valid_start_positions)
                end_loss = nn.CrossEntropyLoss()(end_logits, valid_end_positions)
                span_loss = (start_loss + end_loss) / 2
                # Use the same weighted loss as in training
                pos_weight = torch.tensor([0.3]).to(device)
                answerable_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(answerable_logits, is_impossible)
                loss = span_loss + 0.3 * answerable_loss

                total_val_loss += loss.item()
                val_steps += 1  

                # Get predictions
                start_idx = torch.argmax(start_logits, dim=1)
                end_idx = torch.argmax(end_logits, dim=1)
                # Use the same higher threshold as in prediction
                answerable_pred = (torch.sigmoid(answerable_logits) >= 0.8).long()

                # Store predictions and labels for metrics calculation
                for i, example_id in enumerate(example_ids):
                    all_predictions[example_id] = {
                        'start_idx': start_idx[i].item(),
                        'end_idx': end_idx[i].item(),
                        'is_impossible': answerable_pred[i].item()
                    }
                    all_labels[example_id] = {
                        'start_idx': start_positions[i].item(),
                        'end_idx': end_positions[i].item(),
                        'is_impossible': is_impossible[i].item()
                    }

        # Calculate metrics
        metrics = calculate_metrics(all_predictions, all_labels)
        avg_val_loss = total_val_loss / val_steps

        print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {avg_val_loss:.4f}, EM: {metrics['exact_match']:.4f}, F1: {metrics['f1']:.4f}, Answerable Acc: {metrics['answerable_accuracy']:.4f}")

        # Save model if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_squad_model.pt')
            print(f"Saved best model with Val Loss: {best_val_loss:.4f}")

    # Load best model
    model.load_state_dict(torch.load('best_squad_model.pt'))

    # Example predictions
    print("\nExample predictions:")

    # Example 1
    question1 = "What is the capital of France?"
    context1 = "Paris is the capital and most populous city of France, with an estimated population of 2,175,601 residents as of 2018, in an area of more than 105 square kilometres."
    answer1 = predict(model, processor, question1, context1, device)
    print(f"\nExample 1:")
    print(f"Question: {question1}")
    print(f"Context: {context1}")
    print(f"Answer: {answer1}")

    # Example 2
    question2 = "Who invented the telephone?"
    context2 = "Alexander Graham Bell was a Scottish-born inventor, scientist, and engineer who is credited with inventing and patenting the first practical telephone. He also co-founded the American Telephone and Telegraph Company in 1885."
    answer2 = predict(model, processor, question2, context2, device)
    print(f"\nExample 2:")
    print(f"Question: {question2}")
    print(f"Context: {context2}")
    print(f"Answer: {answer2}")

    # Example 3
    question3 = "What is the boiling point of water?"
    context3 = "Water boils at 100 degrees Celsius at standard atmospheric pressure. At higher altitudes, the boiling point is lower due to decreased atmospheric pressure."
    answer3 = predict(model, processor, question3, context3, device)
    print(f"\nExample 3:")
    print(f"Question: {question3}")
    print(f"Context: {context3}")
    print(f"Answer: {answer3}")

    # Example 4 (unanswerable)
    question4 = "Who is the president of Mars?"
    context4 = "Mars is the fourth planet from the Sun and the second-smallest planet in the Solar System, being larger than only Mercury. In English, Mars carries the name of the Roman god of war."
    answer4 = predict(model, processor, question4, context4, device)
    print(f"\nExample 4 (should be unanswerable):")
    print(f"Question: {question4}")
    print(f"Context: {context4}")
    print(f"Answer: {answer4}")

    print("Script completed successfully!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
