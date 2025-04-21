"""
Collection of prompt templates used in the Boss Assistant application.
Separating prompts makes them easier to manage and edit.
"""

# Prompt template for HR department
HR_PROMPT_TEMPLATE = """
You are an HR assistant. Answer the question based on the context below.
If the question cannot be answered based on the context, say "I don't know".

Context: {context}

Question: {question}

Answer:
"""

# Prompt template for Finance department
FINANCE_PROMPT_TEMPLATE = """
You are a Finance assistant. Answer the question based on the context below.
If the question cannot be answered based on the context, say "I don't know".

Context: {context}

Question: {question}

Answer:
"""

# Prompt template for text translation
TRANSLATION_PROMPT = """
Please translate the following text to {target_language} and correct its grammar.
Only return the translated text, no explanations or additional text.

Text to translate: {text}
"""

# Prompt template for language detection
LANGUAGE_DETECTION_PROMPT = """
Identify the language of the following text. Return only one word: 'english', 'vietnamese', or 'other'.
Do not include any explanations or additional text in your response.

Text: {text}
"""

# Prompt template for step-back question
STEP_BACK_PROMPT = """
Given the following question, generate a more general question that provides context.
The step-back question should be broader and help understand the context of the original question.
Only return the step-back question, no explanations.

Original question: {question}
"""

# Prompt template for rephrasing questions
REPHRASE_PROMPT = """
Rephrase the following question in a different way while maintaining its meaning.
Only return the rephrased question, no explanations.

Original question: {question}
"""

# Prompt template for decomposing questions
DECOMPOSE_PROMPT = """
Break down the following complex question into 2-3 simpler sub-questions.
Each sub-question should address a specific aspect of the original question.
Return only the numbered sub-questions, one per line, no explanations.

Original question: {question}
"""

# Prompt template for semantic routing
ROUTING_PROMPT = """
Analyze the following question and determine which department it belongs to.
Consider the semantic meaning and context, not just keywords.

HR department handles: employee matters, recruitment, training, benefits, compensation,
payroll, leave, vacation, performance reviews, onboarding, offboarding, workplace policies,
employee relations, and staff development.

Finance department handles: financial matters, budgeting, revenue tracking, expense management,
profit analysis, financial reporting, investments, taxes, accounting, audits, fiscal planning,
quarterly reports, annual reports, balance sheets, and business performance.

First, determine the department: 'hr', 'finance', or 'unknown'.
Then, rate your confidence in this decision on a scale of 0 to 1, where:
- 0.9-1.0: Extremely confident, the question is clearly about this department
- 0.7-0.8: Very confident, the question is most likely about this department
- 0.5-0.6: Moderately confident, the question seems related to this department
- 0.3-0.4: Somewhat confident, the question might be related to this department
- 0.0-0.2: Not confident, cannot determine the department

Return your answer in this exact format: "department: [hr/finance/unknown], confidence: [0-1]"

Question: {question}
"""

# Prompt template for answer summarization
SUMMARIZATION_PROMPT = """
Based on the following questions and their corresponding answers, please provide a comprehensive and coherent final answer.
The final answer should:
1. Be well-structured and easy to understand
2. Include all relevant information from the component answers
3. Remove any redundant information
4. Maintain the original meaning and context
5. Be concise while being complete
6. Answer in Vietnamese

Original question: {question}

Component questions and answers:
{context}

Please provide the final comprehensive answer:
"""
