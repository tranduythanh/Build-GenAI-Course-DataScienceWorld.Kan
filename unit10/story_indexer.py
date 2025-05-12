import os
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import (
    Document,
    StorageContext,
    VectorStoreIndex,
    KeywordTableIndex,
    ComposableGraph,
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.retrievers import KeywordTableSimpleRetriever
from collections import defaultdict
import multiprocessing as mp
from tqdm import tqdm
import math
import warnings
import logging

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="llama_index")
logging.getLogger("root").setLevel(logging.ERROR)

# ------------------ Load .env ------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY, "❗ Vui lòng cài đặt OPENAI_API_KEY trong file .env của bạn."

# ------------------ CONFIG ------------------
OPENAI_MODEL = "gpt-4.1-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 10  # Số lượng nodes trong mỗi batch

def process_batch(args):
    nodes, story_name, batch_idx, total_batches, embed_model, llm = args
    
    # Create storage context for this batch
    batch_storage = StorageContext.from_defaults()
    
    # Create indices for this batch
    vector_index = VectorStoreIndex(
        nodes,
        storage_context=batch_storage,
        embed_model=embed_model,
        show_progress=False  # Disable progress bar for each batch
    )
    
    keyword_index = KeywordTableIndex(
        nodes,
        storage_context=batch_storage,
        llm=llm,
        show_progress=False  # Disable progress bar for each batch
    )
    
    return vector_index, keyword_index

def load_and_parse_documents(folder_path: Path):
    story_documents = defaultdict(list)
    story_nodes = defaultdict(list)
    
    for txt_file in folder_path.rglob("*.txt"):
        story_name = txt_file.stem  # Get story name from filename
        with open(txt_file, "r", encoding="utf-8") as f:
            text = f.read()
            doc = Document(
                text=text,
                metadata={
                    "source": txt_file.name,
                    "story_name": story_name
                }
            )
            story_documents[story_name].append(doc)

    parser = SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=50)
    
    for story_name, docs in story_documents.items():
        story_nodes[story_name].extend(parser.get_nodes_from_documents(docs))
    
    return story_documents, story_nodes

def build_or_load_graph_index(story_documents, story_nodes, persist_dir="./storage"):
    print("🔄 Đang khởi tạo embedding model và LLM...")
    embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
    llm = OpenAI(model=OPENAI_MODEL, temperature=0.1, api_key=OPENAI_API_KEY)

    story_indices = {}
    story_summaries = {}
    story_retrievers = {}
    
    # Create main storage directory if it doesn't exist
    os.makedirs(persist_dir, exist_ok=True)
    
    # Always create new indices
    print("🆕 Đang tạo mới chỉ mục cho tất cả truyện...")
    
    # Sử dụng multiprocessing để xử lý song song
    num_processes = min(mp.cpu_count() - 1, 4)  # Giới hạn số processes để tránh quá tải
    print(f"🖥️ Sử dụng {num_processes} processes để xử lý song song")
    
    for story_name, nodes in story_nodes.items():
        print(f"\n📊 Đang xử lý truyện '{story_name}' với {len(nodes)} nodes...")
        story_storage_path = os.path.join(persist_dir, story_name)
        os.makedirs(story_storage_path, exist_ok=True)
        
        # Chia nodes thành các batch
        num_batches = math.ceil(len(nodes) / BATCH_SIZE)
        batches = []
        for i in range(num_batches):
            start_idx = i * BATCH_SIZE
            end_idx = min((i + 1) * BATCH_SIZE, len(nodes))
            batches.append(nodes[start_idx:end_idx])
        
        # Tạo arguments cho multiprocessing
        process_args = [
            (batch, story_name, i, num_batches, embed_model, llm)
            for i, batch in enumerate(batches)
        ]
        
        # Xử lý song song các batch
        with mp.Pool(num_processes) as pool:
            batch_results = list(tqdm(
                pool.imap(process_batch, process_args),
                total=len(batches),
                desc=f"Xử lý {story_name}",
                ncols=100
            ))
        
        # Kết hợp kết quả từ các batch
        vector_indices = []
        keyword_indices = []
        for vector_idx, keyword_idx in batch_results:
            vector_indices.append(vector_idx)
            keyword_indices.append(keyword_idx)
        
        # Lưu trữ indices
        story_indices[story_name] = {
            'vector': vector_indices,
            'keyword': keyword_indices
        }
        story_summaries[story_name] = f"Truyện {story_name}"
        
        # Tạo hybrid retriever cho mỗi batch
        batch_retrievers = []
        for vector_idx, keyword_idx in zip(vector_indices, keyword_indices):
            vector_retriever = VectorIndexRetriever(
                index=vector_idx,
                similarity_top_k=10
            )
            keyword_retriever = KeywordTableSimpleRetriever(
                index=keyword_idx,
                similarity_top_k=10
            )
            
            class BatchHybridRetriever:
                def __init__(self, vector_retriever, keyword_retriever):
                    self.vector_retriever = vector_retriever
                    self.keyword_retriever = keyword_retriever
                    
                def retrieve(self, query_str):
                    vector_nodes = self.vector_retriever.retrieve(query_str)
                    keyword_nodes = self.keyword_retriever.retrieve(query_str)
                    
                    # Combine and deduplicate nodes
                    seen_node_ids = set()
                    combined_nodes = []
                    
                    for node in vector_nodes + keyword_nodes:
                        if node.node.node_id not in seen_node_ids:
                            seen_node_ids.add(node.node.node_id)
                            combined_nodes.append(node)
                    
                    return combined_nodes
            
            batch_retrievers.append(BatchHybridRetriever(vector_retriever, keyword_retriever))
        
        # Tạo hybrid retriever tổng hợp cho toàn bộ truyện
        class StoryHybridRetriever:
            def __init__(self, batch_retrievers):
                self.batch_retrievers = batch_retrievers
                
            def retrieve(self, query_str):
                all_nodes = []
                for retriever in self.batch_retrievers:
                    nodes = retriever.retrieve(query_str)
                    all_nodes.extend(nodes)
                
                # Deduplicate nodes
                seen_node_ids = set()
                unique_nodes = []
                for node in all_nodes:
                    if node.node.node_id not in seen_node_ids:
                        seen_node_ids.add(node.node.node_id)
                        unique_nodes.append(node)
                
                return unique_nodes
        
        story_retrievers[story_name] = StoryHybridRetriever(batch_retrievers)
        
        # Persist the storage context after creating the index
        print(f"✅ Đã tạo và lưu chỉ mục vector và keyword cho truyện '{story_name}'")
    
    print("\n🔄 Đang xây dựng đồ thị kiến thức...")
    # Tạo đồ thị với các chỉ mục truyện riêng biệt
    all_vector_indices = []
    all_summaries = []
    
    # Tạo một index tổng hợp cho mỗi truyện
    for story_name, indices in story_indices.items():
        # Lấy vector index đầu tiên của mỗi truyện làm đại diện
        story_vector_index = indices['vector'][0]
        all_vector_indices.append(story_vector_index)
        all_summaries.append(f"Truyện {story_name}")
    
    graph = ComposableGraph.from_indices(
        indices=all_vector_indices,
        root_index_cls=VectorStoreIndex,
        children_indices=all_vector_indices,
        llm=llm,
        summary="Đồ thị truyện Việt",
        index_summaries=all_summaries
    )
    print("✅ Đã hoàn thành xây dựng đồ thị kiến thức.")
    return graph, story_indices, story_retrievers 