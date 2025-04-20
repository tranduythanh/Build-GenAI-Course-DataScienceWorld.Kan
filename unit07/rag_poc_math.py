# Import các thư viện cần thiết
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
import glob
import argparse

def translate_to_english(text, llm):
    """
    Translate text to English using the language model.
    
    Args:
        text (str): Text to translate
        llm: Language model instance
        
    Returns:
        str: Translated text in English
    """
    if not text.strip():
        return text
        
    # Create translation prompt
    translation_prompt = f"""
    Please translate the following text to English. 
    If the text is already in English, return it as is.
    Only return the translated text, no explanations or additional text.
    
    Text to translate: {text}
    """
    
    try:
        # Get translation
        translated_text = llm.invoke(translation_prompt).content
        return translated_text.strip()
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # Return original text if translation fails

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='RAG system for processing markdown files')
    parser.add_argument('--data-dir', 
                       type=str, 
                       default='data/markdown',
                       help='Directory containing markdown files (default: data/markdown)')
    return parser.parse_args()

def read_markdown_files(directory):
    """
    Read all markdown files from a directory and combine their contents.
    
    Args:
        directory (str): Path to the directory containing markdown files
        
    Returns:
        str: Combined content of all markdown files
    """
    if not os.path.exists(directory):
        raise ValueError(f"Directory {directory} does not exist")
        
    # Get all markdown files in the directory
    markdown_files = glob.glob(os.path.join(directory, "*.md"))
    
    if not markdown_files:
        raise ValueError(f"No markdown files found in {directory}")
        
    combined_content = ""
    for file_path in markdown_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Add a separator between files
            combined_content += f"\n\n# File: {os.path.basename(file_path)}\n\n{content}\n"
            
    return combined_content

def main():
    """Main function to run the RAG system"""
    # Parse command line arguments
    args = parse_arguments()
    markdown_dir = args.data_dir

    # Đọc nội dung từ các file markdown
    math_markdown = read_markdown_files(markdown_dir)
    print(f"Successfully read markdown files from {markdown_dir}")

    # --- BƯỚC 1: Chia nhỏ văn bản Markdown ---
    # Định nghĩa các header để chia nhỏ văn bản
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    # Tạo bộ chia văn bản với các header đã định nghĩa
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    # Chia văn bản thành các đoạn nhỏ dựa trên header
    documents = markdown_splitter.split_text(math_markdown)

    # Tạo bộ chia văn bản dựa trên ký tự
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
    )
    
    # Chia nhỏ các đoạn văn bản thành các chunk nhỏ hơn
    texts = []
    for doc in documents:
        chunks = char_splitter.split_text(doc.page_content)
        texts.extend(chunks)

    # --- BƯỚC 2: Tạo embeddings ---
    # Khởi tạo mô hình embeddings từ OpenAI (yêu cầu OPENAI_API_KEY)
    embedding_model = OpenAIEmbeddings()
    # Tạo vector database từ các đoạn văn bản
    vector_db = FAISS.from_texts(texts, embedding_model)

    # --- BƯỚC 3: Tạo prompt template ---
    # Định nghĩa template cho prompt, yêu cầu trả lời ngắn gọn và rõ ràng bằng tiếng Việt
    prompt_template = """
    Your are a math assistant. Answer closely to the provided relevant information. Use Vietnamese if possible.

    [Relevant information]
    {context}

    [Question]
    {question}
    """

    # Tạo prompt template với các biến đầu vào
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )

    # --- BƯỚC 4: Khởi tạo mô hình ngôn ngữ ---
    # Sử dụng GPT-4 Turbo Preview làm mô hình ngôn ngữ
    llm = ChatOpenAI(model="gpt-4-turbo-preview")

    # --- BƯỚC 5: Tạo RAG Chain ---
    # Tạo chuỗi RAG để trả lời câu hỏi
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,  # Mô hình ngôn ngữ
        chain_type="stuff",  # Loại chuỗi
        retriever=vector_db.as_retriever(search_kwargs={"k": 3}),  # Lấy 3 đoạn văn bản liên quan nhất
        chain_type_kwargs={"prompt": prompt},  # Sử dụng prompt template đã định nghĩa
        verbose=True  # Hiển thị thông tin chi tiết khi chạy
    )

    # --- BƯỚC 6: Interactive Q&A Loop ---
    print("\nWelcome to the Math Assistant!")
    print("You can ask questions about mathematics in any language.")
    print("Type 'exit' or 'quit' to end the session.")
    print("-" * 50)

    while True:
        # Get user input
        question = input("\nYour question: ").strip()
            
        # Skip empty questions
        if not question:
            print("Please enter a question.")
            continue
            
        try:
            # Translate question to English
            print("\nTranslating your question...")
            english_question = translate_to_english(question, llm)
            if english_question != question:
                print(f"Translated question: {english_question}")
            
            # Get and display answer
            print("\nProcessing your question...")
            
            # Get relevant chunks from vector DB
            retriever = vector_db.as_retriever(search_kwargs={"k": 3})
            results = vector_db.similarity_search_with_score(english_question, k=3)
            for doc, score in results:
                print(f"\033[31mScore: {score}\033[0m {doc.page_content}")
            
            # Get the prompt that's being constructed
            prompt_text = prompt.format(
                context="\n".join([doc.page_content for doc, _ in results]),
                question=english_question
            )
            print("\n>> \033[31mPrompt:\033[0m")  # Red color for prompt
            print(f"\033[31m{prompt_text}\033[0m")
            
            answer = qa_chain.invoke({"query": english_question})["result"]
            print("\n>> \033[33mAnswer:\033[0m")  # Yellow color for answer
            print(f"\033[33m{answer}\033[0m")
            print("-" * 50)
        except Exception as e:
            print(f"\nError processing your question: {e}")
            print("Please try again with a different question.")

if __name__ == "__main__":
    main()
