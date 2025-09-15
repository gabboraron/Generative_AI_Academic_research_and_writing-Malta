"""
Basic LLM Usage Examples for Scientific Text Processing

This module demonstrates fundamental operations with Large Language Models
for scientific research applications.
"""

import os
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Example 1: Text Generation for Scientific Writing
def generate_scientific_text(prompt: str, model_name: str = "gpt2") -> str:
    """
    Generate scientific text using a language model.
    
    Args:
        prompt: Input text to continue
        model_name: Name of the model to use
        
    Returns:
        Generated text continuation
    """
    try:
        from transformers import pipeline
        
        generator = pipeline('text-generation', model=model_name)
        result = generator(
            prompt, 
            max_length=200, 
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True
        )
        return result[0]['generated_text']
    
    except ImportError:
        return "Error: transformers library not installed. Run: pip install transformers torch"
    except Exception as e:
        return f"Error generating text: {str(e)}"


# Example 2: Scientific Text Summarization
def summarize_paper(text: str, max_length: int = 130) -> str:
    """
    Summarize scientific text using a pre-trained summarization model.
    
    Args:
        text: Scientific text to summarize
        max_length: Maximum length of summary
        
    Returns:
        Summary of the input text
    """
    try:
        from transformers import pipeline
        
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
        # Truncate text if too long
        if len(text) > 1000:
            text = text[:1000]
            
        summary = summarizer(text, max_length=max_length, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    
    except ImportError:
        return "Error: transformers library not installed. Run: pip install transformers torch"
    except Exception as e:
        return f"Error summarizing text: {str(e)}"


# Example 3: Scientific Entity Recognition
def extract_scientific_entities(text: str) -> List[Dict]:
    """
    Extract named entities from scientific text.
    
    Args:
        text: Scientific text to analyze
        
    Returns:
        List of extracted entities with labels
    """
    try:
        from transformers import pipeline
        
        ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
        entities = ner(text)
        
        # Group consecutive entities
        grouped_entities = []
        current_entity = None
        
        for entity in entities:
            if entity['entity'].startswith('B-'):
                if current_entity:
                    grouped_entities.append(current_entity)
                current_entity = {
                    'text': entity['word'],
                    'label': entity['entity'][2:],
                    'confidence': entity['score']
                }
            elif entity['entity'].startswith('I-') and current_entity:
                current_entity['text'] += ' ' + entity['word']
                current_entity['confidence'] = min(current_entity['confidence'], entity['score'])
        
        if current_entity:
            grouped_entities.append(current_entity)
            
        return grouped_entities
    
    except ImportError:
        return [{"error": "transformers library not installed. Run: pip install transformers torch"}]
    except Exception as e:
        return [{"error": f"Error extracting entities: {str(e)}"}]


# Example 4: Question Answering for Scientific Text
def answer_scientific_question(context: str, question: str) -> str:
    """
    Answer questions about scientific text using a QA model.
    
    Args:
        context: Scientific text containing the answer
        question: Question to answer
        
    Returns:
        Answer extracted from the context
    """
    try:
        from transformers import pipeline
        
        qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
        result = qa_pipeline(question=question, context=context)
        
        return f"Answer: {result['answer']} (Confidence: {result['score']:.3f})"
    
    except ImportError:
        return "Error: transformers library not installed. Run: pip install transformers torch"
    except Exception as e:
        return f"Error answering question: {str(e)}"


# Example 5: Text Classification for Scientific Papers
def classify_scientific_text(text: str, labels: List[str]) -> Dict:
    """
    Classify scientific text into predefined categories.
    
    Args:
        text: Scientific text to classify
        labels: List of possible labels
        
    Returns:
        Classification results with scores
    """
    try:
        from transformers import pipeline
        
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        result = classifier(text, labels)
        
        return {
            'predicted_label': result['labels'][0],
            'all_scores': dict(zip(result['labels'], result['scores']))
        }
    
    except ImportError:
        return {"error": "transformers library not installed. Run: pip install transformers torch"}
    except Exception as e:
        return {"error": f"Error classifying text: {str(e)}"}


# Demo function to showcase all examples
def run_demo():
    """
    Run a demonstration of all basic LLM operations.
    """
    print("=== Basic LLM Usage Examples for Scientific Text ===\n")
    
    # Sample scientific text
    sample_text = """
    Large language models have revolutionized natural language processing by enabling 
    few-shot learning capabilities. Recent studies show that transformer-based architectures 
    can achieve state-of-the-art performance on various scientific text analysis tasks. 
    The attention mechanism allows these models to capture long-range dependencies in 
    scientific documents, making them particularly useful for literature review and 
    hypothesis generation.
    """
    
    print("1. Text Generation:")
    prompt = "The implications of artificial intelligence in scientific research include"
    generated = generate_scientific_text(prompt)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated}\n")
    
    print("2. Text Summarization:")
    summary = summarize_paper(sample_text)
    print(f"Original text length: {len(sample_text)} characters")
    print(f"Summary: {summary}\n")
    
    print("3. Named Entity Recognition:")
    entities = extract_scientific_entities(sample_text)
    print("Extracted entities:")
    for entity in entities[:3]:  # Show first 3 entities
        if 'error' not in entity:
            print(f"  - {entity['text']} ({entity['label']}, confidence: {entity['confidence']:.3f})")
        else:
            print(f"  - {entity['error']}")
    print()
    
    print("4. Question Answering:")
    question = "What allows models to capture long-range dependencies?"
    answer = answer_scientific_question(sample_text, question)
    print(f"Question: {question}")
    print(f"{answer}\n")
    
    print("5. Text Classification:")
    labels = ["computer science", "biology", "physics", "chemistry"]
    classification = classify_scientific_text(sample_text, labels)
    if 'error' not in classification:
        print(f"Predicted category: {classification['predicted_label']}")
        print("All scores:")
        for label, score in classification['all_scores'].items():
            print(f"  - {label}: {score:.3f}")
    else:
        print(f"Error: {classification['error']}")


if __name__ == "__main__":
    run_demo()