# Tools and Utilities

This directory contains scripts, utilities, and tools to support learning and research with LLMs in scientific contexts.

## 🛠️ Available Tools

### Data Processing Tools
- **text_preprocessor.py** - Clean and preprocess scientific text
- **pdf_extractor.py** - Extract text from scientific PDFs
- **citation_parser.py** - Parse and analyze citations
- **metadata_extractor.py** - Extract paper metadata

### Analysis Tools
- **similarity_analyzer.py** - Compare document similarity
- **topic_modeler.py** - Perform topic modeling on scientific texts
- **sentiment_analyzer.py** - Analyze sentiment in research papers
- **keyword_extractor.py** - Extract key terms and concepts

### LLM Integration Tools
- **openai_interface.py** - Wrapper for OpenAI API
- **huggingface_utils.py** - Utilities for Hugging Face models
- **prompt_templates.py** - Templates for scientific prompts
- **batch_processor.py** - Process multiple documents efficiently

### Visualization Tools
- **network_visualizer.py** - Visualize citation networks
- **topic_visualizer.py** - Create topic model visualizations
- **performance_plotter.py** - Plot model performance metrics
- **wordcloud_generator.py** - Generate word clouds from text

## 📦 Installation and Setup

### Requirements
Create a virtual environment and install dependencies:

```bash
# Create virtual environment
python -m venv llm_env
source llm_env/bin/activate  # On Windows: llm_env\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### Configuration
1. Copy `config.template.yaml` to `config.yaml`
2. Add your API keys and configuration settings
3. Adjust paths and parameters as needed

## 🚀 Quick Start Examples

### Extract Text from PDF
```python
from tools.pdf_extractor import PDFExtractor

extractor = PDFExtractor()
text = extractor.extract_text("paper.pdf")
print(text[:500])  # Print first 500 characters
```

### Analyze Paper Similarity
```python
from tools.similarity_analyzer import SimilarityAnalyzer

analyzer = SimilarityAnalyzer()
similarity = analyzer.compare_papers("paper1.txt", "paper2.txt")
print(f"Similarity score: {similarity:.3f}")
```

### Generate Research Summary
```python
from tools.openai_interface import OpenAIInterface

llm = OpenAIInterface()
summary = llm.summarize_paper("paper_text.txt")
print(summary)
```

## 🔧 Tool Categories

### 1. **Preprocessing Tools**
- Text cleaning and normalization
- Format conversion (PDF to text)
- Metadata extraction
- Citation parsing

### 2. **Analysis Tools**
- Statistical text analysis
- Similarity measurements
- Topic modeling
- Sentiment analysis

### 3. **LLM Integration**
- API wrappers and interfaces
- Prompt engineering utilities
- Batch processing tools
- Response parsing

### 4. **Visualization**
- Network analysis plots
- Statistical visualizations
- Interactive dashboards
- Export utilities

## 📝 Usage Guidelines

### Best Practices
1. **Data Privacy**: Never upload sensitive data to external APIs
2. **Rate Limiting**: Respect API rate limits and usage policies
3. **Validation**: Always validate tool outputs before using in research
4. **Documentation**: Keep detailed logs of tool usage and parameters

### Common Workflows

#### Literature Review Workflow
1. Extract text from papers using `pdf_extractor.py`
2. Preprocess text with `text_preprocessor.py`
3. Analyze topics using `topic_modeler.py`
4. Visualize results with `topic_visualizer.py`

#### Paper Analysis Workflow
1. Extract metadata with `metadata_extractor.py`
2. Analyze citations using `citation_parser.py`
3. Calculate similarities with `similarity_analyzer.py`
4. Generate network visualization with `network_visualizer.py`

## 🔐 Security and Privacy

### API Key Management
- Store API keys in environment variables
- Use `.env` files for local development
- Never commit API keys to version control
- Rotate keys regularly

### Data Handling
- Process sensitive data locally when possible
- Use anonymization techniques for shared datasets
- Follow institutional data policies
- Implement proper access controls

## 🧪 Testing

### Running Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test category
python -m pytest tests/test_preprocessing.py

# Run with coverage
python -m pytest --cov=tools tests/
```

### Test Data
- Sample datasets available in `tests/data/`
- Mock API responses for testing
- Validation datasets for accuracy checks

## 🤝 Contributing New Tools

### Tool Development Guidelines
1. Follow the existing code structure and naming conventions
2. Include comprehensive docstrings and type hints
3. Add unit tests for all functions
4. Provide usage examples in docstrings
5. Update this README with tool descriptions

### Submission Process
1. Create a new branch for your tool
2. Implement the tool with tests
3. Update documentation
4. Submit a pull request for review

## 📚 Dependencies

Key libraries used in this toolkit:
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computations
- `scikit-learn` - Machine learning utilities
- `transformers` - Hugging Face model library
- `openai` - OpenAI API client
- `matplotlib/seaborn` - Visualization
- `networkx` - Network analysis
- `pdfplumber` - PDF text extraction

## 🐛 Troubleshooting

### Common Issues
- **Import errors**: Check virtual environment activation
- **API errors**: Verify API keys and rate limits
- **Memory issues**: Use batch processing for large datasets
- **Performance**: Consider using GPU acceleration when available

### Getting Help
- Check the tool documentation and examples
- Review the troubleshooting section in each tool
- Ask questions in the course discussion forum
- Submit issues for bugs or feature requests