import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
import os
from dotenv import load_dotenv
import logging
import re

# Cấu hình logging với định dạng chi tiết
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clean_sql_query(query: str) -> str:
    """
    Làm sạch và trích xuất truy vấn SQL từ phản hồi của LLM.
    Loại bỏ định dạng markdown, dấu backticks và bất kỳ văn bản không phải SQL nào.
    """
    logger.info("="*50)
    logger.info("Raw query before cleaning:")
    logger.info(query)
    logger.info("="*50)
    
    # Loại bỏ các khối mã markdown
    query = re.sub(r'```sql|```', '', query)
    
    # Loại bỏ bất kỳ văn bản nào trước SELECT đầu tiên (không phân biệt chữ hoa/thường)
    query = re.sub(r'^.*?(?=SELECT)', '', query, flags=re.IGNORECASE)
    
    # Loại bỏ bất kỳ văn bản nào sau dấu chấm phẩy cuối cùng
    query = re.sub(r';.*$', ';', query)
    
    # Loại bỏ các câu lệnh action hoặc checker
    query = re.sub(r'Action:.*$', '', query, flags=re.DOTALL)
    query = re.sub(r'Action Input:.*$', '', query, flags=re.DOTALL)
    query = re.sub(r'Now, I will.*$', '', query, flags=re.DOTALL)
    query = re.sub(r'It seems that.*$', '', query, flags=re.DOTALL)
    
    # Loại bỏ khoảng trắng ở đầu/cuối
    query = query.strip()
    
    logger.info("Cleaned query:")
    logger.info(query)
    logger.info("="*50)
    
    return query

class CleanSQLDatabaseToolkit(SQLDatabaseToolkit):
    def get_tools(self):
        """Lấy các công cụ trong toolkit."""
        query_tool = QuerySQLDataBaseTool(
            db=self.db,
            clean_query_fn=clean_sql_query
        )
        return [query_tool]

# Tải các biến môi trường từ file .env
load_dotenv()

# In các biến môi trường để debug
logger.info("Environment Variables:")
for key, value in os.environ.items():
    if "OPENAI" in key or "API" in key:  # Chỉ in các biến môi trường liên quan
        logger.info(f"{key}: {value}")

# Kiểm tra các biến môi trường bắt buộc
if not os.getenv("OPENAI_API_KEY"):
    st.error("Vui lòng cài đặt khóa API OpenAI trong file .env")
    st.stop()

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Trợ lý Cơ sở dữ liệu Chinook",
    page_icon="🎵",
    layout="centered"
)

# Khởi tạo trạng thái phiên cho agent và LLM
if 'agent_executor' not in st.session_state:
    try:
        # Kết nối đến cơ sở dữ liệu Chinook.db
        db = SQLDatabase.from_uri("sqlite:///Chinook.db")
        
        # Tải mô hình ngôn ngữ GPT-4
        llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-4o-mini",
            verbose=True
        )
        
        # Chuẩn bị bộ công cụ với custom toolkit
        toolkit = CleanSQLDatabaseToolkit(db=db, llm=llm)
        
        # Tạo agent với custom system message
        system_message = """You are an expert SQL developer (SQLite). When given a question, you should:
        1. Think about the question carefully
        2. Write a SQL query to answer the question
        3. Return ONLY the SQL query, without any markdown formatting, explanations, or additional text
        """
        
        # Tạo agent với verbose logging
        st.session_state.agent_executor = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            verbose=True,
            system_message=system_message,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )
        
        # Lưu LLM vào session state để sử dụng cho việc dịch
        st.session_state.llm = llm
    except Exception as e:
        st.error(f"Lỗi khi khởi tạo kết nối cơ sở dữ liệu: {str(e)}")
        st.stop()

# Các thành phần giao diện người dùng
st.title("🎵 Trợ lý Cơ sở dữ liệu Chinook")
st.markdown("""
    Hãy hỏi tôi bất cứ điều gì về cửa hàng nhạc! Tôi có thể giúp bạn tìm thông tin về:
    - Khách hàng và các giao dịch mua hàng
    - Nghệ sĩ và album
    - Bài hát và thể loại
    - Doanh số và hóa đơn
""")

# Thêm ô nhập văn bản cho câu hỏi
question = st.text_input(
    "Bạn muốn biết điều gì?",
    placeholder="Ví dụ: Khách hàng nào đã chi tiêu nhiều tiền nhất?"
)

# Thêm nút gửi
submit_button = st.button("Hỏi")

# Xử lý câu hỏi khi được gửi
if submit_button and question:
    try:
        with st.spinner("Đang xử lý..."):
            # Dịch câu hỏi sang tiếng Anh
            translation_prompt = f"""
            Translate the following Vietnamese question to English. 
            Only return the English translation, nothing else.
            Question: {question}
            """
            logger.info("Translating question to English...")
            english_question = st.session_state.llm.invoke(translation_prompt).content
            logger.info(f"Translated question: {english_question}")
            
            # Hiển thị câu hỏi đã dịch
            st.info(f"Đã dịch câu hỏi: {english_question}")
            
            # Thực thi truy vấn SQL với logging
            logger.info("="*50)
            logger.info("Starting SQL query execution...")
            logger.info(f"Original question: {english_question}")
            
            # Lấy cả phản hồi cuối cùng và các bước trung gian
            result = st.session_state.agent_executor.invoke({"input": english_question})
            raw_response = result["output"]
            
            # Ghi log các bước trung gian
            logger.info("\nIntermediate Steps:")
            for step in result.get("intermediate_steps", []):
                action = step[0]
                observation = step[1]
                logger.info(f"Action: {action.tool}")
                logger.info(f"Action Input: {action.tool_input}")
                logger.info(f"Observation: {observation}")
                logger.info("-"*50)
            
            logger.info("Final response:")
            logger.info(raw_response)
            logger.info("="*50)
            
            response = raw_response
            
            # Dịch câu trả lời sang tiếng Việt
            translation_prompt = f"""
            Translate the following English answer to Vietnamese. 
            DO NOT translate any proper nouns, names, song titles, album names, or artist names.
            Keep all numbers, dates, and technical terms in their original form.
            Only return the Vietnamese translation, nothing else.
            Answer: {response}
            """
            vietnamese_response = st.session_state.llm.invoke(translation_prompt).content
            
            st.success("Đây là kết quả tôi tìm được:")
            st.write(vietnamese_response)
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        st.error(f"Đã xảy ra lỗi khi xử lý câu hỏi của bạn: {str(e)}")

# Thêm các câu hỏi mẫu
with st.expander("Câu hỏi mẫu"):
    st.markdown("""
    Hãy thử hỏi những câu như:
    - Nghệ sĩ nào có nhiều album nhất?
    - 5 thể loại nhạc bán chạy nhất là gì?
    - Khách hàng nào đã chi tiêu nhiều tiền nhất trong năm 2013?
    - Có bao nhiêu bài hát thuộc thể loại Rock?
    - Liệt kê tất cả các album của một nghệ sĩ cụ thể
    """)

# Thêm chân trang
st.markdown("---")
st.markdown("Được tạo bằng ❤️ sử dụng Streamlit, LangChain và OpenAI") 