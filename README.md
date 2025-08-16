# 📄 Smart PDF Chatbot  

An **AI-powered chatbot** that allows you to **upload PDF documents** and interact with them through natural language queries.  
Built with **Python, LangChain, FAISS, Streamlit, and OpenAI**, this project demonstrates how to combine **LLMs with document embeddings** to create an intelligent knowledge assistant.  

---

## 🚀 Features  
- 📂 Upload one or multiple PDF files  
- 🔎 Extract and split text into chunks for better retrieval  
- 🧠 Generate embeddings with HuggingFace / OpenAI  
- ⚡ Store and search chunks using **FAISS Vector Database**  
- 💬 Ask questions and get context-aware answers  
- 🖥️ Simple and clean **Streamlit UI**  

---

## 🏗️ Tech Stack  
- **Backend**: Python, FastAPI (for APIs)  
- **NLP/AI**: LangChain, HuggingFace, OpenAI  
- **Database**: FAISS (Vector Store)  
- **Frontend/UI**: Streamlit  
- **File Handling**: PyMuPDF  

---

## 📂 Project Structure  

\`\`\`
smart-pdf-chatbot/
│
├── app/                  
│   ├── pdf_handler.py        # ✅ Extract text from PDF
│   ├── text_splitter.py      # ✅ Split PDF text into chunks
│   ├── embedder.py           # ✅ Generate embeddings + store in FAISS
│   ├── qa_chain.py           # ✅ Connect FAISS + LLM via LangChain RetrievalQA
│   └── utils.py              # ✅ Helper functions (e.g., save/load FAISS index)
│
├── api/
│   └── main.py               # ✅ FastAPI backend (optional) for API routes
│
├── ui/
│   └── index.py                # ✅ Streamlit frontend: upload PDF + chat UI
│
├── data/
│   └── index.faiss           # ✅ Stored FAISS index
│   └── pdf_text.txt          # ✅ Extracted and chunked text (optional)
│
├── .env                      # ✅ OpenAI key stored securely
├── requirements.txt
└── README.md
\`\`\`

---

## ⚙️ Installation & Setup  

### 1️⃣ Clone the Repository  
\`\`\`bash
git clone https://github.com/sayan14banerjee/Smart_PDF_Chatbot.git
cd Smart_PDF_Chatbot
\`\`\`

### 2️⃣ Create a Virtual Environment
\`\`\`bash
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows
\`\`\`

### 3️⃣ Install Dependencies
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 4️⃣ Add API Key
Create a `.env` file in the root folder and add:

\`\`\`ini
OPENAI_API_KEY=your_api_key_here
\`\`\`

### 5️⃣ Run the App
\`\`\`bash
streamlit run ui/app.py
\`\`\`

---

## 📸 Demo Preview
(Add screenshot or gif of your app running here for better presentation)

---

## 🤝 Contribution
Contributions are welcome!

1. Fork the repo 🍴
2. Create a new branch 🌱
3. Make your changes ✨
4. Submit a Pull Request 🚀

---

## 📜 License
This project is licensed under the MIT License – feel free to use and modify.

---

## 👨‍💻 Author
**Sayan Banerjee**  
🔗 [LinkedIn](https://linkedin.com/in/sayan14banerjee) | 🌐 [GitHub](https://github.com/sayan14banerjee)

---

⭐ **If you like this project, don't forget to star the repo!**
