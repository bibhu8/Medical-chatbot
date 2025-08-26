# 🏥 AI-Powered Medical Chatbot

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/framework-Streamlit-red.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/AI-LangChain-green.svg)](https://langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent medical chatbot that provides evidence-based answers to medical questions using **Retrieval-Augmented Generation (RAG)** technology. Built with LangChain, FAISS, and powered by Groq's lightning-fast inference.

![Medical Chatbot Demo](https://via.placeholder.com/800x400/1f2937/ffffff?text=Medical+AI+Assistant+Demo)

## 🌟 Features

- **🔍 Intelligent Search**: Searches through comprehensive medical encyclopedia (12MB+ of medical data)
- **⚡ Lightning Fast**: Powered by Groq's optimized inference for sub-second responses  
- **📚 Evidence-Based**: All answers include source citations with page numbers
- **🎯 Medical-Focused**: Specialized prompts for accurate medical information
- **💊 Safety First**: Built-in medical disclaimers and professional guidance
- **🔒 Secure**: Local vector database with no data leakage concerns
- **🎨 Modern UI**: Clean, professional Streamlit interface

## 🚀 How It Works

### Architecture Overview
```
User Query → Embedding Model → FAISS Vector Search → Retrieved Documents → LLM Processing → Formatted Response
```

1. **Document Processing**: Medical encyclopedia is preprocessed and stored in FAISS vector database
2. **Query Embedding**: User questions are converted to embeddings using `sentence-transformers/all-MiniLM-L6-v2`
3. **Semantic Search**: FAISS finds the 3 most relevant medical document sections
4. **LLM Generation**: Groq's Llama model generates accurate responses based on retrieved context
5. **Response Formatting**: Clean, professional output with source citations

## 🛠️ Tech Stack

- **Frontend**: Streamlit (Web UI)
- **LLM**: Meta Llama 4 Maverick via Groq API
- **Embeddings**: HuggingFace Sentence Transformers
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Framework**: LangChain for RAG pipeline
- **Language**: Python 3.9+

## ⚡ Quick Start

### Prerequisites
- Python 3.9 or higher
- Git
- Groq API key (free at [console.groq.com](https://console.groq.com))

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/medical-chatbot.git
cd medical-chatbot
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv medical_chatbot_env

# Activate virtual environment
# On Linux/Mac:
source medical_chatbot_env/bin/activate
# On Windows:
medical_chatbot_env\Scripts\activate
```

### 3. Install Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
```bash
# Create .env file
touch .env

# Add your Groq API key to .env file:
echo "GROQ_API_KEY=your_groq_api_key_here" >> .env
```

**Get your free Groq API key:**
1. Visit [console.groq.com](https://console.groq.com)
2. Sign up/login
3. Navigate to API Keys
4. Create new API key
5. Copy and paste into `.env` file

### 5. Launch the Application
```bash
streamlit run medibot.py
```

### 6. Access the Chatbot
Open your browser and navigate to: `http://localhost:8501`

## 📖 Usage Examples

### Example Medical Queries:
- *"What is diabetes and what are its symptoms?"*
- *"Explain the treatment options for hypertension"*
- *"What causes pneumonia and how is it diagnosed?"*
- *"Tell me about heart disease prevention"*

### Sample Response:
```
Diabetes mellitus is a chronic disease in which the pancreas no longer produces enough insulin or when cells of the body cannot absorb glucose from the blood...

📚 Sources:
📄 The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf (Page 435)
📄 The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf (Page 434)

⚠️ Disclaimer: This information is for educational purposes only. Always consult with healthcare professionals for medical advice.
```

## 📁 Project Structure

```
medical-chatbot/
├── medibot.py              # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .env                   # Environment variables (create this)
├── data/                  # Medical documents directory
│   └── The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf
├── vectorstore/           # FAISS vector database
│   └── db_faiss/
├── create_memory_for_llm.py    # Vector database creation script
├── connect_memory_with_llm.py  # LLM connection utilities
└── README.md              # This file
```

## ⚙️ Configuration

### Environment Variables (.env file):
```env
# Required
GROQ_API_KEY=your_groq_api_key_here

# Optional (for other models)
HUGGINGFACE_API_TOKEN=your_token_here_optional
MISTRAL_API_KEY=your_mistral_key_if_needed
```

### Customization Options:
- **Model Selection**: Change the Groq model in `medibot.py`
- **Search Results**: Modify `search_kwargs={'k':3}` to return more/fewer sources
- **Temperature**: Adjust `temperature=0.1` for more creative/conservative responses
- **Embedding Model**: Switch embedding models in `get_vectorstore()` function

## 🚀 Performance Optimizations

### Why It's So Fast:
1. **Groq Infrastructure**: Sub-second inference times
2. **Optimized Embeddings**: Lightweight sentence-transformers model
3. **FAISS Vector Search**: Blazing fast similarity search
4. **Smart Caching**: Streamlit caching for vector database
5. **Efficient Retrieval**: Only top 3 most relevant documents processed

### Performance Metrics:
- **Query Response Time**: < 2 seconds average
- **Vector Database Size**: ~50MB (optimized)
- **Memory Usage**: ~500MB RAM
- **Concurrent Users**: Supports multiple simultaneous sessions

## 🔧 Troubleshooting

### Common Issues:

**1. API Key Error:**
```bash
# Make sure your .env file exists and has the correct key
cat .env
# Should show: GROQ_API_KEY=gsk_...
```

**2. Package Installation Issues:**
```bash
# On Linux, you might need additional dependencies:
sudo apt-get update
sudo apt-get install python3-dev build-essential

# Or on Arch Linux:
sudo pacman -S python python-pip base-devel
```

**3. Port Already in Use:**
```bash
# Use a different port:
streamlit run medibot.py --server.port 8502
```

**4. Vector Database Loading Error:**
- Ensure `vectorstore/db_faiss/` directory exists
- If missing, run `python create_memory_for_llm.py`

## 📊 System Requirements

### Minimum Requirements:
- **OS**: Windows 10, macOS 10.14, or Linux
- **Python**: 3.9+
- **RAM**: 4GB
- **Storage**: 2GB free space
- **Internet**: Required for initial model downloads and API calls

### Recommended:
- **RAM**: 8GB+ for better performance
- **Storage**: SSD for faster loading
- **Internet**: Stable connection for optimal API response times

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit your changes**: `git commit -m 'Add amazing feature'`
4. **Push to the branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Development Setup:
```bash
# Clone your fork
git clone https://github.com/your-username/medical-chatbot.git

# Create development branch
git checkout -b dev

# Install development dependencies
pip install -r requirements.txt

# Make your changes and test
streamlit run medibot.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain**: For the excellent RAG framework
- **Groq**: For providing fast and reliable AI inference
- **HuggingFace**: For the embedding models and transformers
- **Streamlit**: For the beautiful web framework
- **FAISS**: For efficient similarity search
- **Medical Encyclopedia**: Gale Encyclopedia of Medicine (Second Edition)

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/medical-chatbot/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/medical-chatbot/discussions)
- **Documentation**: See this README and code comments

## 🔮 Future Enhancements

- [ ] Support for multiple medical databases
- [ ] Voice input/output capabilities
- [ ] Multi-language support
- [ ] Mobile app version
- [ ] Integration with medical APIs
- [ ] Advanced medical terminology search
- [ ] User session management
- [ ] Conversation history export

---

**⚠️ Important Medical Disclaimer**: This AI chatbot is for educational and informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified healthcare providers with questions regarding medical conditions.

---

Made with ❤️ and ☕ by [Bibhuti]



