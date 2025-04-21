# Import required libraries
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
import signal
import sys

# Import prompt templates from prompts.py
from prompts import (
    HR_PROMPT_TEMPLATE,
    FINANCE_PROMPT_TEMPLATE,
    SUMMARIZATION_PROMPT
)

# Import functions from custom modules
from text_processing import (
    translate_text,
    step_back_question,
    rephrase_question,
    decompose_question
)

from data_processing import (
    read_data_files,
    create_vector_db,
    load_keywords_from_file
)

from routing import (
    analyze_semantic_routing,
    get_answer_from_department
)

CHAT_GPT_MODEL = "gpt-4o-mini"


def create_qa_chain(vector_db, llm, prompt_template):
    """
    Create QA chain to answer questions.
    Args:
        vector_db: Vector database
        llm: Language model instance
        prompt_template: Prompt template
        chain_name: Chain name for tracking
    Returns:
        RetrievalQA: QA chain
    """
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": prompt},
        verbose=True
    )

    return qa_chain





# Load keywords
hr_keywords = load_keywords_from_file('hr_keywords.txt')
finance_keywords = load_keywords_from_file('finance_keywords.txt')

# Print loaded keywords for verification
print(f"Loaded {len(hr_keywords)} HR keywords and {len(finance_keywords)} Finance keywords")



def process_question(question, llm, hr_qa_chain, finance_qa_chain):
    """
    Process question: translate to English if needed, apply step-back, rephrase, and decomposition strategies,
    then synthesize component answers into a comprehensive final answer.
    Args:
        question (str): Question to process
        llm: Language model instance
        hr_qa_chain: QA chain for HR department
        finance_qa_chain: QA chain for Finance department
    Returns:
        str: Processed and synthesized answer
    """
    print("\n===== QUESTION PROCESSING PIPELINE =====\n")

    # Check for exit commands
    if question.lower() in ['exit', 'quit', 'thoát', 'thoat', 'q']:
        return "EXIT_COMMAND"

    # Translate the original question
    question = translate_text(question, llm)
    print(f"\n[Translated question]: {question}")

    # Determine main department for the original question
    main_department, main_confidence = analyze_semantic_routing(question, llm, hr_keywords, finance_keywords)
    print(f"\n[Main department]: {main_department} (confidence: {main_confidence:.2f})")

    # Apply question processing strategies
    step_back = step_back_question(question, llm)
    print(f"\n[Step-back question]: {step_back}")

    rephrased = rephrase_question(question, llm)
    print(f"\n[Rephrased question]: {rephrased}")

    decomposed = decompose_question(question, llm)
    print("\n[Sub-questions]:")
    for i, sub_q in enumerate(decomposed, 1):
        print(f"  {i}. {sub_q}")

    # Create list of sub-questions to process (not including original question)
    questions_to_process = [step_back, rephrased] + decomposed

    # Process each sub-question and store corresponding question-answer pairs
    print("\n===== COMPONENT ANSWERS =====\n")
    qa_pairs = []
    for i, q in enumerate(questions_to_process, 1):
        if q.strip():
            print(f"\n[Processing question {i}]: {q}")
            department, confidence = analyze_semantic_routing(q, llm, hr_keywords, finance_keywords)
            print(f"  - Department: {department} (confidence: {confidence:.2f})")
            answer, _ = get_answer_from_department(q, department, main_department, hr_qa_chain, finance_qa_chain)
            print(f"  - Answer: {answer}")
            if answer and answer.lower() != "i don't know":
                qa_pairs.append((q, answer))
                print(f"  - Added to valid answers list")
            else:
                print(f"  - Skipping this answer (no information)")

    # If no valid answers or all are "I don't know"
    print(f"\n[Number of valid answers]: {len(qa_pairs)}")
    if not qa_pairs:
        print("\n[No valid answers, trying direct question]")
        # Try asking the original question directly
        direct_answer, _ = get_answer_from_department(question, main_department, main_department, hr_qa_chain, finance_qa_chain)
        print(f"\n[Direct answer]: {direct_answer}")
        if direct_answer and direct_answer.lower() != "i don't know":
            print("\n[Using direct answer]")
            return direct_answer
        print("\n[No relevant information found]")
        return "Sorry, I couldn't find any information relevant to your question."

    # Create context for summarization
    context = "\n\n".join([f"Question: {q}\nAnswer: {a}" for q, a in qa_pairs])
    print("\n===== ANSWER SUMMARIZATION =====\n")
    print(f"[Summarization context]:\n{context}")

    # Create prompt for summarization
    summarization_prompt = SUMMARIZATION_PROMPT.format(question=question, context=context)

    # Generate final summarized answer
    try:
        print("\n[Generating final answer]")
        final_answer = llm.invoke(summarization_prompt).content.strip()
        print(f"\n[Final answer]:\n{final_answer}")
        return final_answer
    except Exception as e:
        print(f"Error during summarization: {e}")
        # Fallback: return combined answers if summarization fails
        fallback = "\n\n".join([a for _, a in qa_pairs])
        print(f"\n[Fallback answer]:\n{fallback}")
        return fallback

def signal_handler(sig, frame):
    """
    Handle interrupt signal (Ctrl+C).
    Args:
        sig: Signal received
        frame: Current frame
    """
    print("\n\nExiting Boss Assistant. Goodbye!")
    sys.exit(0)

def show_help():
    """
    Display help information.
    """
    print("\n=== Boss Assistant Help ===")
    print("- Type your question in English or Vietnamese")
    print("- Type 'exit', 'quit', or 'q' to exit the program")
    print("- Type 'help' or '?' to show this help message")
    print("- When asked to choose a department, enter:")
    print("  1: HR Department")
    print("  2: Finance Department")
    print("===========================\n")

def main():
    # Set up signal handler for interrupt signal (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)

    # Disable warnings
    import warnings
    # Disable all warnings
    warnings.filterwarnings("ignore")

    # Set up OpenAI API key
    if "OPENAI_API_KEY" not in os.environ:
        print("Error: OPENAI_API_KEY is not set.")
        print("Please set the OPENAI_API_KEY environment variable.")
        return

    # Initialize language model
    llm = ChatOpenAI(model=CHAT_GPT_MODEL)

    # Read data from files
    hr_data, finance_data = read_data_files()

    # Create embedding model
    embedding_model = OpenAIEmbeddings()

    # Create vector databases for HR and Finance
    hr_vector_db = create_vector_db(hr_data, embedding_model)
    finance_vector_db = create_vector_db(finance_data, embedding_model)

    # Use prompt templates from prompts.py
    hr_prompt_template = HR_PROMPT_TEMPLATE
    finance_prompt_template = FINANCE_PROMPT_TEMPLATE

    # Create QA chains for HR and Finance
    hr_qa_chain = create_qa_chain(hr_vector_db, llm, hr_prompt_template)
    finance_qa_chain = create_qa_chain(finance_vector_db, llm, finance_prompt_template)

    print("Welcome to Boss Assistant!")
    print("You can ask questions about HR and Finance.")
    print("Type 'help' or '?' for assistance, or 'exit' to quit.")
    print("-" * 50)

    while True:
        try:
            question = input("Your question: ")

            if not question:
                print("Please enter a question.")
                continue

            if question.lower() in ['help', '?', 'trợ giúp', 'tro giup']:
                show_help()
                continue

            # Process question and get answer
            answer = process_question(question, llm, hr_qa_chain, finance_qa_chain)

            # Check if user wants to exit
            if answer == "EXIT_COMMAND":
                print("\nExiting Boss Assistant. Goodbye!")
                break

            print("\n>> Answer:")
            print(answer)

            print("-" * 50)

        except KeyboardInterrupt:
            print("\n\nExiting Boss Assistant. Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.")

if __name__ == "__main__":
    main()
