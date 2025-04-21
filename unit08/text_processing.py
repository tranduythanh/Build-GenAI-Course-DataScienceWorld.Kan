"""
Text processing utilities for the Boss Assistant application.
Contains functions for translation, step-back questioning, rephrasing, and decomposition.
"""

from prompts import (
    TRANSLATION_PROMPT,
    STEP_BACK_PROMPT,
    REPHRASE_PROMPT,
    DECOMPOSE_PROMPT
)

def translate_text(text, llm, target_language="English"):
    """
    Translate text to the target language using the language model.
    Args:
        text (str): Text to translate
        llm: Language model instance
        target_language (str): Target language for translation
    Returns:
        str: Translated text
    """
    if not text.strip():
        return text

    # Always translate the text
    translation_prompt = TRANSLATION_PROMPT.format(target_language=target_language, text=text)

    try:
        print("\n[Translation Request]")
        print(f"Original text: {text}")
        translated_text = llm.invoke(translation_prompt).content
        print(f"Translated text: {translated_text.strip()}")
        return translated_text.strip()
    except Exception as e:
        print(f"Translation error: {e}")
        return text

def step_back_question(question, llm):
    """
    Use step-back prompting technique to get a more general context for the question.
    Args:
        question (str): Original question
        llm: Language model instance
    Returns:
        str: Step-back question
    """
    step_back_prompt = STEP_BACK_PROMPT.format(question=question)

    try:
        print("\n[Step-back Question Request]")
        print(f"Original question: {question}")
        step_back = llm.invoke(step_back_prompt).content.strip()
        print(f"Step-back question: {step_back}")
        return step_back
    except Exception as e:
        print(f"Step-back error: {e}")
        return question

def rephrase_question(question, llm):
    """
    Rephrase the question to make it clearer and more specific.
    Args:
        question (str): Original question
        llm: Language model instance
    Returns:
        str: Rephrased question
    """
    rephrase_prompt = REPHRASE_PROMPT.format(question=question)

    try:
        print("\n[Rephrase Question Request]")
        print(f"Original question: {question}")
        rephrased = llm.invoke(rephrase_prompt).content.strip()
        print(f"Rephrased question: {rephrased}")
        return rephrased
    except Exception as e:
        print(f"Rephrase error: {e}")
        return question

def decompose_question(question, llm):
    """
    Decompose complex question into simpler sub-questions.
    Args:
        question (str): Original question
        llm: Language model instance
    Returns:
        list: List of sub-questions
    """
    decompose_prompt = DECOMPOSE_PROMPT.format(question=question)

    try:
        print("\n[Question Decomposition Request]")
        print(f"Original question: {question}")
        decomposed = llm.invoke(decompose_prompt).content.strip()
        sub_questions = [q.strip() for q in decomposed.split('\n') if q.strip()]
        print("Sub-questions:")
        for i, q in enumerate(sub_questions, 1):
            print(f"{i}. {q}")
        return sub_questions
    except Exception as e:
        print(f"Decompose error: {e}")
        return [question]
