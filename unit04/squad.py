import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
from sklearn.model_selection import train_test_split

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
            """Simple whitespace tokenization"""
        return text.split()

    def convert_tokens_to_ids(self, tokens):
            """Convert tokens to ids using vocabulary"""
        return [self.vocab.get(token, self.vocab['[UNK]']) for token in tokens]

    def read_squad_examples(self, file_path):
            """Read SQuAD examples from file"""
        print(f"Reading SQuAD examples from {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
        print(f"Reading SQuAD examples from {file_path}...")
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
        print(f"Building vocabulary from {len(all_texts)} texts...")
        self.build_vocab(all_texts)

        print(f"Loaded {len(examples)} examples from {file_path}")
        print(f"Vocabulary size: {self.vocab_size}")
        return examples

    def convert_examples_to_features(self, examples):
        """Convert examples to features that can be fed to the model"""
        print("Converting examples to print("C...")
o       features nverting examples to features...")
                features = []
        i, numerate(e)
        for if i % 1000 == 0:
                print(f"Processing example {i}/{len(examples)}")

            i, example in enumerate(examples):
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
                af padditg_length > 0:
                    intention_mask = [1] * len(input_ids)

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
                features.append(feat, max_examples=1000ure)
         with a limit on examples
        print(f"Converted {len(examples)} examples to {len(features)} features")

        # Limit the number or extmples for fasuer processing
        if max_examples > 0 and max_examples < len(examples):
            print(f"Limiting to {max_examples} examples for faster processing")
            examples = examples[:max_examples]

        featrn features

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
    def __init__(self, vocab_size, d_model=256, num_heads=8, d_ff=512, num_layers=4, dropout_rate=0.1):
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

                # Get encoder outputs
        sequence_output = self.bert_encoder(input_ids, attention_mask)

        # Apply dropout
                sequence_output = self.dropout(sequence_output)

        # Get logits for start and end positions
  M   gits = self.qa_outputs(sequence_output)
    m  so):
eo1("Strg SQuAD 2t0 Qu   goi Answ#rloggsciiwt..."l classification (using [CLS] token)
    equence_output[:, 0, :]
    # Set swvec a(GPUgifslv, sebla,rws CPU)
unctfor.cda._aalabl(:
main():dvceor('cuda'
t("SQlAf0torch. Qe An sim._alabl(eaic Uirch.ilbk ots.mew.is_buel:
if torchdevacilab(or):('mps'orch.bedApsleiSiohcm  2M1/M2))
  elelse:
se:device=orch.evce('c')
ce =prich(f"Uegnd vdce: {evce}")
    # Process data (limit to 1000 examples for faster processing)
essor ProQrsua aimt1000 dxamp ettfo  fadael drictg
n_fetr,ce voral_SQuADPu ce= sr(s
tedsxmps,eu Qa,sv)cvocabz=prcssor.prce_data('dv-v2.0.jrn',dmaxenxtmpait=1000)    print(f"Validation dataset size: {len(val_dataset)}")

Sptdatainttn advid sets
ateata odrfeur,rva _featureDat(trann_t_satspbit(featurei, zest_8ize=0.1,rnim_etote=42

modeBECrQserndgscs_ize)
    cdatSQuADDaatpratn_fzatues
    vdataetSQuADDas(vfaguo b)tch to test...")

    Traidaastiz: aast
    pri t(f"V devicedata iizt:d{]vn(icetst}")

    toCeeatpsdata load=r h['token_type_ids'].to(device)
    ttaarrlp]dero=cDa)Ladratastbatch_size=8, shusfoe=True)
na  ch['en'der.=dDataLoader(vicedsebtchsize=8        is_impossible = batch['is_impossible'].to(device)

    Initiliz
   prnt("Initizgdl w#thFoocrburary pize:s{ocbize}")
  stmodela=rBERTForQu_loionAnswering(ioctb,dizelgocibsnize)swerable_logits = model(
    nput_io(dv

    # Intni_lizakoptinizor
_pimzer = pm.A  cWe loss.parmes()lr=3-5
    art and end positions
    # Train f r jus  1 b tch tr te_t = nn.CrossEntropyLoss()(start_logits, start_positions)
    enlns("Trains=g.for 1 batch to tnlt...")
   #mFdelatrainleclassification

    answerable_lo r=in nn.BCEWithLogitsLoss()(answerable_logits, is_impossible)

    # Combine losses
    loss = span_loss + answerable_loss

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
  opti)
    print(f"Batch loss: {loss.item():.4f}")
    break  # Just one batch for testing

print("Script completed successfully!")
)
_name__     try:
    main()
        # For start and end positions
except Exception as e:
    import traceback
    print(f"Error: {e}")
   trac(
Fl=BEWLogLCbnwbBcwzz.zpp"Bh{4kJ"Scy!