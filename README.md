# 📚 GitHub Documentation Helper

An AI-powered documentation assistant that helps you quickly find answers from your README files and project documentation using RAG (Retrieval-Augmented Generation).

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![LangChain](https://img.shields.io/badge/LangChain-Latest-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🌟 Features

- 📁 **Multi-file Support** - Upload multiple README.md or documentation files
- 🔍 **Smart Search** - Uses FAISS vector similarity search for accurate results
- 🤖 **AI-Powered Answers** - Leverages Groq's Mixtral model for intelligent responses
- 💬 **Interactive Chat** - Natural conversation interface with chat history
- 📊 **Statistics Dashboard** - Track files indexed, chunks created, and questions asked
- 🎨 **Beautiful UI** - Modern gradient design with smooth animations
- 📄 **Source Citations** - Shows which files were used to answer your questions

## 🎯 Use Cases

- Quickly understand new GitHub projects
- Find installation instructions instantly
- Get configuration details without reading entire docs
- Understand project features and usage
- Onboard new developers faster
- Create documentation FAQs

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **LLM Framework**: LangChain
- **Vector Store**: FAISS (Facebook AI Similarity Search)
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **LLM**: Groq API (Mixtral-8x7b-32768)
- **Language**: Python 3.8+

## 📋 Prerequisites

- Python 3.8 or higher
- Groq API key (free at [console.groq.com](https://console.groq.com))

## 🚀 Installation

### Step 1: Clone or Create Project Directory

```bash
mkdir github-doc-helper
cd github-doc-helper
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install streamlit langchain langchain-groq langchain-community langchain-core faiss-cpu sentence-transformers python-dotenv
```

Or use requirements.txt:

```bash
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables

Create a `.env` file in your project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

**Get your Groq API key:**
1. Go to [console.groq.com](https://console.groq.com)
2. Sign up (free tier available)
3. Navigate to API Keys
4. Create and copy your key

### Step 5: Create app.py

Copy the application code into `app.py` (see installation guide or repository).

## 🎮 Usage

### Running the Application

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

### Using the App

1. **Enter API Key** - Paste your Groq API key in the sidebar (or use .env file)
2. **Upload Files** - Select your README.md or .txt documentation files
3. **Process Documents** - Click "🚀 Process & Index Documents"
4. **Ask Questions** - Type your questions in the chat interface

### Example Questions

```
"How do I install this project?"
"What are the main features?"
"Show me usage examples"
"What dependencies are required?"
"How do I configure the settings?"
"What API endpoints are available?"
```

## 📁 Project Structure

```
github-doc-helper/
├── venv/                  # Virtual environment (auto-generated)
├── .env                   # Environment variables (create this)
├── .gitignore            # Git ignore file
├── app.py                # Main application
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GROQ_API_KEY` | Your Groq API key | Yes |

### Customization Options

You can customize these parameters in `app.py`:

```python
# Chunk size for text splitting
chunk_size=1000
chunk_overlap=200

# Number of relevant chunks to retrieve
search_kwargs={"k": 3}

# LLM temperature (0-1, lower = more focused)
temperature=0.3

# Embedding model
model_name="sentence-transformers/all-MiniLM-L6-v2"
```

## 📊 How It Works

### RAG (Retrieval-Augmented Generation) Pipeline

```
1. Document Upload
   └─> Upload README.md or .txt files

2. Text Processing
   └─> Split documents into chunks
   └─> Create embeddings using Sentence Transformers
   └─> Store in FAISS vector database

3. Query Processing
   └─> User asks a question
   └─> Convert question to embedding
   └─> Search for similar chunks in FAISS
   └─> Retrieve top 3 most relevant chunks

4. Answer Generation
   └─> Send chunks + question to Groq LLM
   └─> Generate contextual answer
   └─> Display with source citations
```

## 🎨 UI Features

- **Gradient Theme** - Beautiful purple/blue gradient design
- **Progress Indicators** - Visual feedback during processing
- **Statistics Dashboard** - Real-time metrics
- **Chat History** - Persistent conversation view
- **Source Citations** - Know where answers come from
- **Responsive Layout** - Works on desktop and tablet
- **Expandable Sections** - Help, tech stack, and file details

## 🐛 Troubleshooting

### Common Issues

**Import Errors**
```bash
# Solution: Clean reinstall
pip uninstall langchain langchain-community langchain-groq langchain-core -y
pip install langchain langchain-groq langchain-community langchain-core
```

**API Key Not Working**
- Verify key at console.groq.com
- Check .env file has no spaces: `GROQ_API_KEY=gsk_...`
- Try entering key directly in app instead

**Slow First Run**
- First processing downloads embedding model (~80MB)
- Subsequent runs are much faster
- This is normal behavior

**FAISS Issues on M1/M2 Mac**
```bash
# Use conda instead
conda install -c pytorch faiss-cpu
```

## 📦 Requirements.txt

```
streamlit
langchain
langchain-groq
langchain-community
langchain-core
faiss-cpu
sentence-transformers
python-dotenv
```

## 🚦 Performance

- **Processing Speed**: ~1-2 seconds per document
- **Query Response**: ~2-3 seconds average
- **Supported File Sizes**: Up to 10MB per file
- **Concurrent Users**: Single user (local deployment)

## 🔐 Security Notes

- Never commit `.env` file to git
- Keep your API key private
- Use environment variables for sensitive data
- The app runs locally - your data stays on your machine

## 📝 Example Use Case

```python
# Sample README.md content
"""
# My Awesome Project

## Installation
pip install my-project

## Features
- Fast processing
- Easy API
- Great docs

## Usage
from my_project import MyClass
obj = MyClass()
result = obj.process()
"""

# Questions you can ask:
- "How do I install this?"
  → Answer: "You can install using: pip install my-project"
  
- "What are the features?"
  → Answer: "This project offers: Fast processing, Easy API, Great docs"
```

## 🌐 Deployment Options

### Local Development (Current)
```bash
streamlit run app.py
```

### Streamlit Cloud (Free)
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository
4. Add GROQ_API_KEY in secrets

### Docker (Advanced)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py"]
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://python.langchain.com/) - LLM framework
- [Groq](https://groq.com/) - Fast LLM inference
- [FAISS](https://github.com/facebookresearch/faiss) - Vector similarity search
- [Streamlit](https://streamlit.io/) - Web framework
- [Sentence Transformers](https://www.sbert.net/) - Embeddings

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

## 🎯 Roadmap

- [ ] Add support for PDF files
- [ ] Implement conversation memory
- [ ] Add export chat history feature
- [ ] Support for multiple languages
- [ ] Add local LLM support (Ollama)
- [ ] Implement document comparison
- [ ] Add API endpoint mode
- [ ] Docker compose setup

## 📚 Additional Resources

- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [Groq API Docs](https://console.groq.com/docs)
- [FAISS Guide](https://github.com/facebookresearch/faiss/wiki)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

**Made with ❤️ using LangChain, FAISS, Groq API, and Streamlit**

⭐ Star this repo if you find it helpful!