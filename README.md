# Trợ lý Agentic RAG Tiếng Việt

Trợ lý AI sử dụng mô hình RAG (Retrieval-Augmented Generation) để trả lời câu hỏi bằng tiếng Việt, kết hợp **kiến thức từ tài liệu nội bộ** và **tìm kiếm web thời gian thực**.

---


## Demo
- Video demo: https://youtu.be/z8Tgwy7uaXw

---
## Tính năng chính

- Trích xuất dữ liệu từ `.txt`, `.docx`, `.pdf`, `.csv`, `.xlsx`
- Xử lý & tạo vector embedding bằng `intfloat/multilingual-e5-base`
- Lưu trữ & tìm kiếm thông tin qua FAISS
- Tìm kiếm thông tin thời gian thực qua Tavily API
- Tương tác với người dùng qua giao diện Gradio
- Mô hình ngôn ngữ chính: `gemini-2.0-flash` (Google Generative AI)
- Trả lời tiếng Việt tự nhiên

---

##  Kiến trúc hệ thống
````
┌─────────────┐
│ Người dùng  │
└──────┬──────┘
       ↓
┌───────────────┐
│ Gradio UI     │
└──────┬────────┘
       ↓
┌────────────────────┐
│ LangChain Agent    │
│ (Zero-shot ReAct)  │
└──────┬─────────────┘
       ↓
┌──────────────┬──────────────┐
│ FAISS DB     │ Tavily API   │
│ (nội bộ)     │ (internet)   │
└──────────────┴──────────────┘
       ↓
┌──────────────────┐
│ Gemini 2.0 Flash │
└──────┬───────────┘
       ↓
┌─────────────────────┐
│ Phản hồi người dùng │
└─────────────────────┘

````
---

````
 Cấu trúc thư mục

.
├── agent.py                # Mã nguồn chính
├── dataset/                # Thư mục chứa tài liệu người dùng
│   └── vectorstores/       # FAISS vectorstore
└── requirements.txt        # (Tùy chọn) Danh sách thư viện cần thiết


````

## Cách chạy project

### 1. Cài đặt thư viện

```bash
pip install -r requirements.txt
````

**Hoặc thủ công:**

```bash
pip install langchain langchain-community langchain-google-genai \
            langchain-huggingface transformers faiss-cpu \
            pandas python-docx PyPDF2 gradio tqdm tavily
```

### 2. Đặt API Key

Tạo file `.env` (khuyến khích) hoặc chỉnh trực tiếp trong `agent.py`:

```bash
TAVILY_API_KEY=your_tavily_key
GOOGLE_API_KEY=your_google_api_key
```

### 3. Thêm tài liệu vào thư mục `dataset/`

Hỗ trợ: `.txt`, `.docx`, `.pdf`, `.csv`, `.xlsx`

### 4. Chạy ứng dụng

```bash
python agent.py
```

Truy cập đường link Gradio để bắt đầu chat với trợ lý AI.

---

## Ví dụ câu hỏi

* "Tìm thông tin về thuế thu nhập cá nhân năm 2024"
* "Tìm các bài viết liên quan đến trí tuệ nhân tạo gần đây"

---

## Đóng góp

Pull request hoặc issue được hoan nghênh!
