# 📚 Chatbot Truyện Việt (Graph-powered)

Một chatbot thông minh sử dụng RAG (Retrieval-Augmented Generation) và đồ thị kiến thức để trả lời các câu hỏi về truyện Việt Nam.

## 🏗️ Cấu trúc dự án

```
.
├── main.py              # File chính chứa toàn bộ logic của chatbot
├── books/              # Thư mục chứa các file truyện (.txt)
├── storage/           # Thư mục lưu trữ các chỉ mục vector
└── .env               # File chứa các biến môi trường (OPENAI_API_KEY)
```

## 🚀 Tính năng chính

- **Xử lý văn bản thông minh**: Sử dụng LlamaIndex để phân tích và xử lý các file truyện
- **Đồ thị kiến thức**: Tạo và quản lý đồ thị kiến thức giữa các truyện
- **Giao diện chat**: Sử dụng Gradio để tạo giao diện chat thân thiện
- **Tìm kiếm thông minh**: Hỗ trợ tìm kiếm trong một truyện cụ thể hoặc tất cả truyện

## ⚙️ Cài đặt

1. Cài đặt các thư viện cần thiết:
```bash
pip install llama-index gradio python-dotenv
```

2. Tạo file `.env` và thêm API key của OpenAI:
```
OPENAI_API_KEY=your_api_key_here
```

3. Tạo thư mục `books` và thêm các file truyện (.txt) vào đó

## 🎯 Cách sử dụng

1. Chạy file main.py:
```bash
python main.py
```

2. Truy cập giao diện web được tạo bởi Gradio (thường là http://localhost:7860)

3. Nhập câu hỏi và chọn truyện muốn tìm kiếm (hoặc "Tất cả truyện")

## 🔧 Cấu hình

Các thông số có thể điều chỉnh trong `main.py`:

- `BOOKS_DIR`: Thư mục chứa các file truyện
- `INDEX_PATH`: Thư mục lưu trữ chỉ mục
- `OPENAI_MODEL`: Model GPT được sử dụng (mặc định: "gpt-4.1-mini")
- `EMBEDDING_MODEL`: Model embedding được sử dụng (mặc định: "text-embedding-3-small")

## 📝 Ví dụ sử dụng

- "Tóm tắt nội dung truyện này?"
- "Nhân vật chính là ai?"
- "Ý nghĩa ẩn dụ trong truyện là gì?"

## 🔄 Quy trình xử lý

1. **Đọc và phân tích tài liệu**:
   - Đọc các file truyện từ thư mục `books`
   - Phân tích thành các node nhỏ hơn

2. **Xây dựng chỉ mục**:
   - Tạo vector index cho từng truyện
   - Lưu trữ chỉ mục vào thư mục `storage`

3. **Tạo đồ thị kiến thức**:
   - Kết nối các chỉ mục truyện thành một đồ thị
   - Cho phép tìm kiếm thông minh giữa các truyện

4. **Xử lý câu hỏi**:
   - Phân tích câu hỏi người dùng
   - Tìm kiếm thông tin liên quan
   - Tạo câu trả lời dựa trên ngữ cảnh tìm được 