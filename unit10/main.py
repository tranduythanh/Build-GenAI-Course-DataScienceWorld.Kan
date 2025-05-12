# story_rag_chatbot_graph.py

import os
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import gradio as gr
from story_indexer import (
    load_and_parse_documents,
    build_or_load_graph_index,
)

# ------------------ Load .env ------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY, "❗ Vui lòng cài đặt OPENAI_API_KEY trong file .env của bạn."

# ------------------ CONFIG ------------------
OPENAI_MODEL = "gpt-4.1-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

# Khởi tạo LLM và embedding model
llm = OpenAI(model=OPENAI_MODEL)
embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL)
Settings.llm = llm
Settings.embed_model = embed_model

# ------------------ EXAMPLES ------------------
EXAMPLES = [
    "Kể tóm tắt câu chuyện",
    "Ai là nhân vật chính trong truyện?",
    "Kết thúc của câu chuyện như thế nào?",
    "Có những nhân vật nào trong truyện?",
    "Nội dung chính của câu chuyện là gì?"
]



PROMPT = '''
Dựa trên các đoạn văn sau đây, hãy trả lời câu hỏi của người dùng một cách ngắn gọn và chính xác.
    
Câu hỏi: {user_input}

Các đoạn văn liên quan:
{context}

Câu trả lời:
'''


def process_query(user_input, story_retrievers, selected_story, llm):
    if not user_input.strip():
        return "Vui lòng nhập câu hỏi của bạn."
    
    # Nếu chọn "Tất cả truyện", tìm kiếm trên tất cả
    if selected_story == "Tất cả truyện":
        all_nodes = []
        for story_name, retriever in story_retrievers.items():
            nodes = retriever.retrieve(user_input)
            all_nodes.extend(nodes)
    else:
        # Tìm kiếm chỉ trong truyện được chọn
        if selected_story not in story_retrievers:
            return "❌ Không tìm thấy truyện được chọn."
        nodes = story_retrievers[selected_story].retrieve(user_input)
        all_nodes = nodes
        
    if not all_nodes:
        return "❌ Không tìm thấy thông tin liên quan."
        
    # Sắp xếp nodes theo độ liên quan
    all_nodes.sort(key=lambda x: x.score if hasattr(x, 'score') else 0, reverse=True)
    
    # Lấy top 5 nodes
    top_nodes = all_nodes[:5]
    
    # Chuẩn bị context cho LLM
    context = "\n\n".join([
        f"Đoạn văn từ truyện '{node.node.metadata.get('story_name', 'Unknown')}':\n{node.node.text}"
        for node in top_nodes
    ])
    
    # Gọi LLM để tổng hợp câu trả lời
    response = llm.complete(PROMPT.format(user_input=user_input, context=context))
    answer = response.text
    
    # Format kết quả
    
    result = f"=================="
    result += f"\n\n      DEBUG\n"
    result += f"=================="
    result += "📖 Các đoạn văn tham khảo:\n\n"
    for i, node in enumerate(top_nodes, 1):
        result += f"{i}. Đoạn văn từ truyện '{node.node.metadata.get('story_name', 'Unknown')}':\n"
        result += f"   {node.node.text[:200]}...\n"
        if hasattr(node, 'score'):
            result += f"   Độ liên quan: {node.score:.2f}\n\n"

    result += f"\n\n=================="
    result += f"\n\n  FINAL RESPONSE\n"
    result += f"==================\n"
    result += f"📚 Câu trả lời:\n{answer}\n\n"
    
    return result


def create_gradio_interface(story_retrievers, examples, llm):
    # Tạo danh sách truyện cho dropdown
    story_options = ["Tất cả truyện"] + list(story_retrievers.keys())
    
    with gr.Blocks(title="Story RAG Chatbot") as interface:
        gr.Markdown("# 📚 Story RAG Chatbot")
        gr.Markdown("Hỏi đáp về nội dung các truyện đã được index.")
        
        with gr.Row():
            with gr.Column():
                story_dropdown = gr.Dropdown(
                    label="Chọn truyện",
                    choices=story_options,
                    value="Tất cả truyện"
                )
                user_input = gr.Textbox(
                    label="Câu hỏi của bạn",
                    placeholder="Nhập câu hỏi của bạn ở đây...",
                    lines=3
                )
                submit_btn = gr.Button("Gửi câu hỏi")
            
            with gr.Column():
                output = gr.Textbox(
                    label="Kết quả",
                    lines=10,
                    interactive=False
                )
        
        # Thêm examples
        gr.Examples(
            examples=examples,
            inputs=user_input
        )
        
        submit_btn.click(
            fn=lambda x, y: process_query(x, story_retrievers, y, llm),
            inputs=[user_input, story_dropdown],
            outputs=output
        )
        
        user_input.submit(
            fn=lambda x, y: process_query(x, story_retrievers, y, llm),
            inputs=[user_input, story_dropdown],
            outputs=output
        )
    
    return interface


def main():
    # Đường dẫn đến thư mục chứa các file truyện
    books_dir = Path("books")
    
    print(f"🚀 Đang đọc và phân tích các file truyện từ: {books_dir}")
    story_documents, story_nodes = load_and_parse_documents(books_dir)
    
    print(f"📄 Đã tạo {sum(len(nodes) for nodes in story_nodes.values())} node từ {len(story_nodes)} truyện.")
    
    print("🔎 Đang xây dựng hoặc tải hệ thống chỉ mục dạng đồ thị...")
    graph, story_indices, story_retrievers = build_or_load_graph_index(story_documents, story_nodes)
    
    print("🌐 Đang khởi động giao diện web...")
    interface = create_gradio_interface(story_retrievers, EXAMPLES, llm)
    interface.launch(share=False)  # Chỉ mở local server

if __name__ == "__main__":
    main()