"""
Test New NLP Agents (21-25)
"""
from veda.agents.nlp.keyword_extractor import KeywordExtractorAgent
from veda.agents.nlp.language_detector import LanguageDetectorAgent
from veda.agents.nlp.translation import TranslationAgent
from veda.agents.nlp.question_answering import QuestionAnsweringAgent
from veda.agents.nlp.text_generator import TextGeneratorAgent
import json

def test_new_nlp_agents():
    print("\n" + "="*60)
    print("TESTING NEW NLP AGENTS (21-25)")
    print("="*60)
    
    sample_text = """
    Apple Inc. announced record-breaking quarterly earnings today in Cupertino, California.
    CEO Tim Cook expressed optimism about the company's future growth in emerging markets.
    The tech giant reported revenue of $95 billion, exceeding analyst expectations.
    """
    
    # Agent 21: Keyword Extractor
    print("\n[1/5] Keyword Extractor Agent...")
    try:
        extractor = KeywordExtractorAgent()
        keyword_result = extractor.execute({"text": sample_text, "num_keywords": 5})
        print(f"Keywords: {keyword_result.get('keywords', [])}")
        print(f"Key Phrases: {keyword_result.get('key_phrases', [])}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Agent 22: Language Detector
    print("\n[2/5] Language Detector Agent...")
    try:
        detector = LanguageDetectorAgent()
        lang_result = detector.execute({"text": sample_text})
        print(f"Language: {lang_result.get('language', 'unknown')}")
        print(f"Confidence: {lang_result.get('confidence', 0)}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Agent 23: Translation
    print("\n[3/5] Translation Agent...")
    try:
        translator = TranslationAgent()
        trans_result = translator.execute({
            "text": "Hello, how are you?",
            "target_language": "Spanish"
        })
        print(f"Translated: {trans_result.get('translated_text', '')}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Agent 24: Question Answering
    print("\n[4/5] Question Answering Agent...")
    try:
        qa = QuestionAnsweringAgent()
        qa_result = qa.execute({
            "context": sample_text,
            "question": "What was Apple's quarterly revenue?"
        })
        print(f"Answer: {qa_result.get('answer', '')}")
        print(f"Confidence: {qa_result.get('confidence', 0)}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Agent 25: Text Generator
    print("\n[5/5] Text Generator Agent...")
    try:
        generator = TextGeneratorAgent()
        gen_result = generator.execute({
            "prompt": "Write a brief product description for a smartwatch",
            "style": "professional",
            "max_length": 50
        })
        print(f"Generated: {gen_result.get('generated_text', '')[:100]}...")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    print("\n" + "="*60)
    print("NEW NLP AGENTS TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    test_new_nlp_agents()