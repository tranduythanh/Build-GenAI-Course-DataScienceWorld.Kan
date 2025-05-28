# Khai báo các target phony (không tạo file thực tế)
# Giúp make hiểu rằng đây là các lệnh thực thi chứ không phải tên file
.PHONY: help install neo4j-start neo4j-stop neo4j-status neo4j-reset test-connection run streamlit clean build-index

# Target mặc định khi chạy lệnh 'make' không có tham số
# Hiển thị menu hướng dẫn sử dụng các lệnh có sẵn
help:
	@echo "GraphRAG Application Commands"
	@echo "============================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  install          Install Python dependencies"
	@echo "  neo4j-start      Start Neo4j Docker container"
	@echo "  neo4j-stop       Stop Neo4j Docker container"
	@echo "  neo4j-status     Show Neo4j status"
	@echo "  neo4j-reset      Reset Neo4j database (removes all data)"
	@echo ""
	@echo "Application Commands:"
	@echo "  test             Run all unit tests"
	@echo "  test-connection  Test Neo4j connection"
	@echo "  build-index      Build the search index"
	@echo "  run              Run the GraphRAG application (terminal)"
	@echo "  streamlit        Run the Streamlit web interface"
	@echo ""
	@echo "Maintenance Commands:"
	@echo "  clean            Clean up Docker containers and volumes"
	@echo ""
	@echo "Quick Start:"
	@echo "  make install && make neo4j-start && make test-connection && make build-index && make run"

# Cài đặt các thư viện Python cần thiết
# Đọc file requirements.txt và cài đặt tất cả dependencies được liệt kê
install:
	pip install -r requirements.txt

# ============= QUẢN LÝ NEO4J DATABASE =============
# Các lệnh quản lý Neo4j thông qua script setup_neo4j.sh

# Khởi động Neo4j Docker container
# Tạo và chạy container Neo4j với cấu hình từ docker-compose
neo4j-start:
	./setup_neo4j.sh start

# Dừng Neo4j Docker container
# Tạm dừng container mà không xóa dữ liệu
neo4j-stop:
	./setup_neo4j.sh stop

# Kiểm tra trạng thái của Neo4j container
# Hiển thị thông tin container có đang chạy hay không
neo4j-status:
	./setup_neo4j.sh status

# Reset hoàn toàn Neo4j database
# Xóa tất cả dữ liệu và khởi tạo lại database từ đầu
neo4j-reset:
	./setup_neo4j.sh reset

# ============= CÁC LỆNH ỨNG DỤNG =============

# Chạy tất cả unit tests
# Tìm và thực thi tất cả file test trong thư mục tests/
test:
	cd tests && python -m unittest discover -v

# Kiểm tra kết nối đến Neo4j database
# Chỉ test connection mà không chạy toàn bộ test suite
test-connection:
	cd tests && python test_neo4j_connection.py --connection-only

# Chạy ứng dụng GraphRAG chính
# Sử dụng Streamlit với cấu hình tắt auto-reload và file watcher
run:
	streamlit run app.py --server.runOnSave false --server.fileWatcherType none

# Xây dựng search index cho hệ thống
# Tạo các chỉ mục tìm kiếm cần thiết trong database
build-index:
	python build_index.py

# ============= BẢO TRÌ VÀ DỌN DẸP =============

# Dọn dẹp Docker containers và volumes
# Xóa tất cả containers, volumes và system cache để giải phóng dung lượng
clean:
	docker-compose down -v
	docker system prune -f

# ============= CÁC LỆNH TỔNG HỢP =============

# Thiết lập hoàn chỉnh hệ thống từ đầu
# Chạy tuần tự: cài đặt dependencies -> khởi động Neo4j -> test kết nối
setup: install neo4j-start test-connection
	@echo "Setup complete! You can now run: make test && make run"

# Khởi động lại Neo4j một cách nhanh chóng
# Dừng -> khởi động -> test kết nối Neo4j
restart: neo4j-stop neo4j-start test-connection
	@echo "Neo4j restarted successfully!"