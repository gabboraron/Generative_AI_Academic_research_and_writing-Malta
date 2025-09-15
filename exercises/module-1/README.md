# Module 1 Exercises: Foundations of LLMs

This directory contains hands-on exercises for Module 1: Foundations of Large Language Models.

## Exercise 1.1: Understanding Model Architectures

### Objective
Compare the behavior and outputs of different transformer architectures on scientific text.

### Tasks
1. **Model Comparison Study**
   - Use Hugging Face to load GPT-2, BERT, and SciBERT
   - Test each model on the same scientific abstract
   - Compare their outputs and analyze differences

2. **Architecture Analysis**
   - Examine the number of parameters in each model
   - Compare training objectives (MLM vs. autoregressive)
   - Document the trade-offs between different approaches

### Deliverables
- Jupyter notebook with model comparisons
- Written analysis of observed differences
- Recommendations for different use cases

---

## Exercise 1.2: Attention Mechanism Exploration

### Objective
Visualize and understand how attention mechanisms work in transformer models.

### Tasks
1. **Attention Visualization**
   - Use the `bertviz` library to visualize attention patterns
   - Test on scientific sentences with complex dependencies
   - Identify what the model focuses on

2. **Pattern Analysis**
   - Compare attention patterns across different layers
   - Analyze how attention changes with sentence complexity
   - Document interesting patterns you observe

### Deliverables
- Interactive visualizations of attention patterns
- Analysis report on attention behavior
- Examples of successful and failed attention patterns

---

## Exercise 1.3: Scientific Text Classification

### Objective
Build a simple classifier to categorize scientific papers by discipline.

### Tasks
1. **Data Preparation**
   - Collect abstracts from different scientific fields
   - Create a balanced dataset with labels
   - Split into training/validation/test sets

2. **Model Development**
   - Fine-tune a pre-trained model (e.g., DistilBERT)
   - Experiment with different hyperparameters
   - Evaluate performance using appropriate metrics

3. **Comparison Study**
   - Compare general models vs. scientific models (SciBERT)
   - Test with different amounts of training data
   - Analyze what features the model learns

### Deliverables
- Trained classification model
- Performance evaluation report
- Code for data preprocessing and training

---

## Exercise 1.4: Prompt Engineering for Scientific Tasks

### Objective
Learn to design effective prompts for scientific applications.

### Tasks
1. **Prompt Design**
   - Create prompts for scientific paper summarization
   - Design prompts for extracting key findings
   - Develop prompts for generating research questions

2. **Evaluation and Iteration**
   - Test prompts with different language models
   - Evaluate output quality using multiple criteria
   - Iterate and improve prompt designs

### Deliverables
- Collection of effective prompts for scientific tasks
- Evaluation results and analysis
- Best practices guide for scientific prompt engineering

---

## Setup Instructions

### Required Libraries
```bash
pip install transformers torch pandas numpy matplotlib seaborn
pip install bertviz jupyter ipywidgets
pip install scikit-learn datasets evaluate
```

### Data Sources
- **Scientific Abstracts**: Use the arXiv dataset or PubMed abstracts
- **Pre-trained Models**: Available through Hugging Face Model Hub
- **Evaluation Datasets**: Use SciERC, SciFact, or similar benchmarks

### Environment Setup
1. Create a virtual environment for the exercises
2. Install required dependencies
3. Download necessary datasets and models
4. Set up Jupyter notebook environment

---

## Evaluation Criteria

### Technical Implementation (40%)
- Code quality and documentation
- Proper use of libraries and frameworks
- Reproducibility of results
- Error handling and edge cases

### Analysis and Insights (40%)
- Depth of analysis and understanding
- Quality of comparisons and evaluations
- Identification of patterns and trends
- Critical thinking about results

### Communication (20%)
- Clear written explanations
- Effective visualizations
- Proper documentation
- Professional presentation

---

## Submission Guidelines

### Format
- Submit as Jupyter notebooks with clear documentation
- Include all necessary code and data files
- Provide a summary report for each exercise

### Deadlines
- Exercise 1.1: End of Week 1
- Exercise 1.2: End of Week 1
- Exercise 1.3: End of Week 2
- Exercise 1.4: End of Week 2

### Collaboration Policy
- Individual work required for core exercises
- Collaboration encouraged for discussion and debugging
- Clearly cite any external resources used

---

## Additional Challenges (Optional)

### Advanced Exercise 1: Custom Tokenizer
Create a custom tokenizer optimized for scientific text:
- Analyze scientific vocabulary patterns
- Design domain-specific tokenization rules
- Compare with existing tokenizers

### Advanced Exercise 2: Model Interpretability
Deep dive into model interpretability:
- Use SHAP or LIME for feature importance
- Analyze model decision-making process
- Identify potential biases in scientific text processing

### Advanced Exercise 3: Multilingual Scientific Text
Explore multilingual aspects of scientific LLMs:
- Test models on non-English scientific texts
- Analyze cross-lingual transfer capabilities
- Develop multilingual scientific applications

---

## Resources and Support

### Documentation
- [Hugging Face Transformers Guide](https://huggingface.co/docs/transformers/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

### Troubleshooting
- Check the course discussion forum for common issues
- Refer to the tools directory for utility functions
- Contact instructors during office hours

### Example Code
See the `examples/basic_llm_usage/` directory for starter code and demonstrations.