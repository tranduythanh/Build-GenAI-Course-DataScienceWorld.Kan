"""
Routing logic for the Boss Assistant application.
Contains functions for determining which department should handle a question.
"""
import re
from prompts import ROUTING_PROMPT
from data_processing import calculate_keyword_confidence

# Dictionary to store user department preferences with context
user_department_preferences = {}

# Dictionary to store topic clusters
topic_clusters = {}

def get_llm_routing_confidence(question, llm):
    """
    Use LLM to determine department and confidence.
    Args:
        question (str): Question to analyze
        llm: Language model instance
    Returns:
        tuple: (department, confidence)
    """
    routing_prompt = ROUTING_PROMPT.format(question=question)

    try:
        response = llm.invoke(routing_prompt).content.strip().lower()

        # Extract department and confidence from response
        department = 'unknown'
        confidence = 0.0

        if 'department:' in response and 'confidence:' in response:
            # Extract department
            dept_match = re.search(r'department:\s*(hr|finance|unknown)', response)
            if dept_match:
                department = dept_match.group(1)

            # Extract confidence
            conf_match = re.search(r'confidence:\s*([0-9]\.[0-9]|[01])', response)
            if conf_match:
                confidence = float(conf_match.group(1))

        return department, confidence
    except Exception as e:
        print(f"LLM routing error: {e}")
        return 'unknown', 0.0

def analyze_semantic_routing(question, llm, hr_keywords, finance_keywords):
    """
    Analyze question semantics to determine appropriate department.
    Uses multiple strategies and voting system to make a decision.
    Args:
        question (str): Question to analyze
        llm: Language model instance
        hr_keywords: List of HR keywords
        finance_keywords: List of Finance keywords
    Returns:
        tuple: (department, confidence)
    """
    print("\n[Semantic Routing Analysis]")
    print(f"Question to analyze: {question}")

    # Strategy 1: Check keywords
    hr_confidence, hr_matched = calculate_keyword_confidence(question, hr_keywords)
    finance_confidence, finance_matched = calculate_keyword_confidence(question, finance_keywords)

    print(f"Keyword analysis:")
    if hr_matched:
        print(f"  - HR keywords: {', '.join(hr_matched)} (confidence: {hr_confidence:.2f})")
    if finance_matched:
        print(f"  - Finance keywords: {', '.join(finance_matched)} (confidence: {finance_confidence:.2f})")

    # Strategy 2: Use LLM for semantic analysis
    llm_department, llm_confidence = get_llm_routing_confidence(question, llm)
    print(f"LLM analysis: {llm_department} (confidence: {llm_confidence:.2f})")

    # Strategy 3: Check history for similar questions
    history_department = None
    history_confidence = 0.0
    question_hash = hash(question.lower())

    if question_hash in user_department_preferences:
        history_department = user_department_preferences[question_hash]['department']
        history_confidence = 0.8  # High confidence in history
        print(f"History analysis: {history_department} (confidence: {history_confidence:.2f})")

    # Combine results from all strategies
    hr_total_confidence = 0
    finance_total_confidence = 0

    # Add points from keywords
    hr_total_confidence += hr_confidence
    finance_total_confidence += finance_confidence

    # Add points from LLM
    if llm_department == 'hr':
        hr_total_confidence += llm_confidence
    elif llm_department == 'finance':
        finance_total_confidence += llm_confidence

    # Add points from history
    if history_department == 'hr':
        hr_total_confidence += history_confidence
    elif history_department == 'finance':
        finance_total_confidence += history_confidence

    # Determine final department
    final_department = 'unknown'
    final_confidence = 0.0

    if hr_total_confidence > finance_total_confidence:
        final_department = 'hr'
        final_confidence = hr_total_confidence / 3  # Divide by number of strategies
    elif finance_total_confidence > hr_total_confidence:
        final_department = 'finance'
        final_confidence = finance_total_confidence / 3
    else:
        # If equal, use LLM result
        final_department = llm_department
        final_confidence = llm_confidence

    # Limit maximum confidence to 1.0
    final_confidence = min(1.0, final_confidence)

    print(f"Final routing decision: {final_department} (confidence: {final_confidence:.2f})")

    # Save result to history
    user_department_preferences[question_hash] = {
        'department': final_department,
        'confidence': final_confidence,
        'question': question
    }

    return final_department, final_confidence

def get_answer_from_department(question, department, main_department, hr_qa_chain, finance_qa_chain):
    """
    Get answer from the appropriate department based on semantic analysis.
    Args:
        question (str): Question to answer
        department (str): Department determined for the question
        main_department (str): Main department of the original question
        hr_qa_chain: QA chain for HR department
        finance_qa_chain: QA chain for Finance department
    Returns:
        tuple: (str, str) - (Answer from the corresponding department, Selected department)
    """
    global user_department_preferences

    # Check if we have a question hash in our preferences
    question_hash = hash(question.lower())
    if question_hash in user_department_preferences and isinstance(user_department_preferences[question_hash], dict):
        preferred_dept = user_department_preferences[question_hash]['department']
        print(f"Using remembered department preference: {preferred_dept}")
        department = preferred_dept

    if department == 'hr':
        print(f"\n[HR QA Chain Request]")
        print(f"Question: {question}")
        result = hr_qa_chain.invoke({"query": question})["result"]
        print(f"HR Assistant response: {result}")
        return result, 'hr'
    elif department == 'finance':
        print(f"\n[Finance QA Chain Request]")
        print(f"Question: {question}")
        result = finance_qa_chain.invoke({"query": question})["result"]
        print(f"Finance Assistant response: {result}")
        return result, 'finance'
    else:
        # If department can't be determined, use main department
        if main_department == 'hr':
            print(f"\n[HR QA Chain Request (fallback to main department)]")
            print(f"Question: {question}")
            result = hr_qa_chain.invoke({"query": question})["result"]
            print(f"HR Assistant response: {result}")
            return result, 'hr'
        elif main_department == 'finance':
            print(f"\n[Finance QA Chain Request (fallback to main department)]")
            print(f"Question: {question}")
            result = finance_qa_chain.invoke({"query": question})["result"]
            print(f"Finance Assistant response: {result}")
            return result, 'finance'
        else:
            # If still can't determine, query both departments and combine results
            print(f"\n[Querying both departments]")
            print(f"Question: {question}")

            # Query HR department
            print(f"\n[HR QA Chain Request (dual query)]")
            hr_result = hr_qa_chain.invoke({"query": question})["result"]
            print(f"HR Assistant response: {hr_result}")

            # Query Finance department
            print(f"\n[Finance QA Chain Request (dual query)]")
            finance_result = finance_qa_chain.invoke({"query": question})["result"]
            print(f"Finance Assistant response: {finance_result}")

            # Check if results are useful
            hr_useful = hr_result.lower() != "i don't know"
            finance_useful = finance_result.lower() != "i don't know"

            if hr_useful and not finance_useful:
                print(f"\n[Using HR response (finance had no information)]")
                return hr_result, 'hr'
            elif finance_useful and not hr_useful:
                print(f"\n[Using Finance response (HR had no information)]")
                return finance_result, 'finance'
            elif hr_useful and finance_useful:
                # Combine both answers
                print(f"\n[Combining responses from both departments]")
                combined_result = f"HR perspective: {hr_result}\n\nFinance perspective: {finance_result}"
                return combined_result, 'both'
            else:
                # Neither has information
                print(f"\n[Neither department has relevant information]")
                return "I don't have enough information to answer this question.", 'unknown'


