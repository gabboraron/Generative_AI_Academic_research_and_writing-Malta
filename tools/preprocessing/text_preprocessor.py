"""
Text Preprocessing Tools for Scientific Documents

This module provides utilities for cleaning and preprocessing scientific text
for use with Large Language Models.
"""

import re
import string
from typing import List, Dict, Optional, Tuple
import unicodedata


class ScientificTextPreprocessor:
    """
    A comprehensive text preprocessor designed for scientific documents.
    """
    
    def __init__(self, 
                 remove_equations: bool = True,
                 remove_citations: bool = False,
                 normalize_whitespace: bool = True,
                 remove_urls: bool = True):
        """
        Initialize the preprocessor with configuration options.
        
        Args:
            remove_equations: Whether to remove mathematical equations
            remove_citations: Whether to remove citation markers
            normalize_whitespace: Whether to normalize whitespace
            remove_urls: Whether to remove URLs
        """
        self.remove_equations = remove_equations
        self.remove_citations = remove_citations
        self.normalize_whitespace = normalize_whitespace
        self.remove_urls = remove_urls
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile commonly used regex patterns."""
        # Mathematical equations (LaTeX style)
        self.equation_pattern = re.compile(r'\$[^$]*\$|\\\[[^\]]*\\\]|\\\([^)]*\\\)')
        
        # Citations in various formats
        self.citation_patterns = [
            re.compile(r'\[[0-9,\s\-]+\]'),  # [1], [1,2], [1-3]
            re.compile(r'\([^)]*\d{4}[^)]*\)'),  # (Author, 2020)
            re.compile(r'\([\w\s,]+,?\s*\d{4}\)'),  # (Smith et al., 2020)
        ]
        
        # URLs and DOIs
        self.url_pattern = re.compile(r'https?://[^\s]+|www\.[^\s]+|doi:[^\s]+')
        
        # Multiple whitespace
        self.whitespace_pattern = re.compile(r'\s+')
        
        # Figure and table references
        self.figure_ref_pattern = re.compile(r'(Figure|Fig\.?|Table|Tab\.?)\s+\d+', re.IGNORECASE)
        
        # Section headers (simple detection)
        self.section_pattern = re.compile(r'^\d+\.?\s+[A-Z][^.]*$', re.MULTILINE)
    
    def clean_text(self, text: str) -> str:
        """
        Apply all preprocessing steps to the input text.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Cleaned and preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKC', text)
        
        # Remove equations if requested
        if self.remove_equations:
            text = self.equation_pattern.sub(' [EQUATION] ', text)
        
        # Remove URLs if requested
        if self.remove_urls:
            text = self.url_pattern.sub(' [URL] ', text)
        
        # Remove citations if requested
        if self.remove_citations:
            for pattern in self.citation_patterns:
                text = pattern.sub(' [CITATION] ', text)
        
        # Normalize whitespace if requested
        if self.normalize_whitespace:
            text = self.whitespace_pattern.sub(' ', text)
        
        # Remove extra punctuation and clean up
        text = self._clean_punctuation(text)
        
        return text.strip()
    
    def _clean_punctuation(self, text: str) -> str:
        """Clean up punctuation and special characters."""
        # Remove multiple consecutive punctuation marks
        text = re.sub(r'[.]{2,}', '.', text)
        text = re.sub(r'[,]{2,}', ',', text)
        text = re.sub(r'[;]{2,}', ';', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,:;!?])', r'\1', text)
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        
        return text
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """
        Extract different sections from a scientific paper.
        
        Args:
            text: Full paper text
            
        Returns:
            Dictionary with section names as keys and content as values
        """
        sections = {}
        
        # Common section headers in scientific papers
        section_headers = [
            r'abstract',
            r'introduction',
            r'methodology?|methods?',
            r'results?',
            r'discussion',
            r'conclusions?',
            r'references?|bibliography',
            r'acknowledgments?'
        ]
        
        text_lower = text.lower()
        
        for header in section_headers:
            pattern = re.compile(rf'^(\d+\.?\s*)?{header}(\s|$)', re.MULTILINE | re.IGNORECASE)
            matches = list(pattern.finditer(text))
            
            if matches:
                start = matches[0].end()
                # Find the next section or end of text
                next_section = None
                for other_header in section_headers:
                    if other_header != header:
                        next_pattern = re.compile(rf'^(\d+\.?\s*)?{other_header}(\s|$)', 
                                                re.MULTILINE | re.IGNORECASE)
                        next_matches = [m for m in next_pattern.finditer(text) if m.start() > start]
                        if next_matches:
                            if next_section is None or next_matches[0].start() < next_section:
                                next_section = next_matches[0].start()
                
                end = next_section if next_section else len(text)
                section_content = text[start:end].strip()
                
                if section_content:
                    sections[header.replace('?', '')] = self.clean_text(section_content)
        
        return sections
    
    def extract_sentences(self, text: str) -> List[str]:
        """
        Extract individual sentences from text.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting (can be improved with NLTK or spaCy)
        sentences = re.split(r'[.!?]+\s+', text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Filter out very short sentences
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def extract_keywords(self, text: str, max_keywords: int = 20) -> List[str]:
        """
        Extract potential keywords from scientific text.
        
        Args:
            text: Input text
            max_keywords: Maximum number of keywords to return
            
        Returns:
            List of potential keywords
        """
        # Remove common stop words (simple approach)
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'this', 'that', 'these', 'those', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do',
            'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'can', 'also', 'however', 'therefore', 'thus', 'furthermore'
        }
        
        # Extract words (simple tokenization)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter stop words and count frequencies
        word_freq = {}
        for word in words:
            if word not in stop_words and len(word) > 2:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:max_keywords]]


def preprocess_scientific_paper(text: str, 
                               preserve_structure: bool = True) -> Dict[str, str]:
    """
    Convenience function to preprocess a scientific paper.
    
    Args:
        text: Raw paper text
        preserve_structure: Whether to extract sections separately
        
    Returns:
        Dictionary with processed text and metadata
    """
    preprocessor = ScientificTextPreprocessor()
    
    result = {
        'cleaned_text': preprocessor.clean_text(text),
        'keywords': preprocessor.extract_keywords(text),
        'sentences': preprocessor.extract_sentences(text)
    }
    
    if preserve_structure:
        result['sections'] = preprocessor.extract_sections(text)
    
    return result


# Example usage
if __name__ == "__main__":
    # Sample scientific text
    sample_text = r"""
    1. Introduction
    
    Large language models (LLMs) have revolutionized natural language processing [1,2]. 
    These models, such as GPT-3 (Brown et al., 2020), demonstrate remarkable capabilities 
    in few-shot learning. The attention mechanism, first introduced in https://arxiv.org/abs/1706.03762,
    allows models to capture long-range dependencies.
    
    The mathematical formulation can be expressed as: $Attention(Q,K,V) = softmax(QK^T/\sqrt{d_k})V$
    
    2. Methodology
    
    Our approach builds upon previous work... Figure 1 shows the architecture.
    """
    
    print("=== Scientific Text Preprocessing Demo ===\n")
    
    # Initialize preprocessor
    preprocessor = ScientificTextPreprocessor()
    
    print("Original text:")
    print(sample_text)
    print("\n" + "="*50 + "\n")
    
    # Clean the text
    cleaned = preprocessor.clean_text(sample_text)
    print("Cleaned text:")
    print(cleaned)
    print("\n" + "="*50 + "\n")
    
    # Extract sections
    sections = preprocessor.extract_sections(sample_text)
    print("Extracted sections:")
    for section_name, content in sections.items():
        print(f"\n{section_name.upper()}:")
        print(content[:200] + "..." if len(content) > 200 else content)
    
    print("\n" + "="*50 + "\n")
    
    # Extract keywords
    keywords = preprocessor.extract_keywords(sample_text)
    print("Extracted keywords:")
    print(", ".join(keywords[:10]))
    
    print("\n" + "="*50 + "\n")
    
    # Use convenience function
    result = preprocess_scientific_paper(sample_text)
    print("Using convenience function:")
    print(f"Number of sentences: {len(result['sentences'])}")
    print(f"Number of sections: {len(result['sections'])}")
    print(f"Top keywords: {', '.join(result['keywords'][:5])}")