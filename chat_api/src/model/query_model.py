from groq import Groq

PROMPT_SYSTEM_REWRITE = """You are a query optimization assistant. Your task is to refine the user’s input into a concise, unambiguous, and context-rich query optimized for similarity search in a RAG-based chatbot.
## Instructions:
- Condense: Remove filler words while retaining all key context from the conversation history.
- Clarify: Resolve pronouns, vague terms, or ambiguous references using chat history.
- Enrich: Add implicit context (e.g., timeframes, entities, intent) if missing but critical.
- Focus: Eliminate redundancy. Prioritize information directly relevant to retrieval.
- Expand: Generate semantically similar queries or paraphrases of the original query to capture different expressions of the same intent.
- Decompose: Break down complex queries into simpler sub-queries, each targeting a distinct aspect of the original question.
- Abstract: Translate specific queries into broader concepts to aid in generalization and retrieval.
- Structure: Output only the optimized queries in the format of the following JSON example below.

## JSON-Format
'{"optimized_queries": ["This is example query 1!", 
                        "This is example query 2!", 
                        "And this is example query 3!]}

'

Return ONLY the JSON-object. Do NOT include extra text.
"""


PROMPT_USER = """Improve the user query using the previous chat history if relevant.
                
## Query: {user_query}

## Instructions:
- Generate multiple optimized queries by applying techniques such as expansion, decomposition, and abstraction.
- Ensure each query is concise, unambiguous, and context-rich.
- Return only the optimized queries in the specified JSON-format"""







class QueryModel():

    def __init__(self, path, api_key):
        self.model = Groq(api_key=api_key)
        self.path = path


    def opt_query(self, query, chat_history=None, MAX_TOKENS=128):

        message = [{"role": "system",
                    "content": PROMPT_SYSTEM_REWRITE}]

        if chat_history:
            message.extend(chat_history)


        message.append({"role": "user",
                        "content": PROMPT_USER.format(user_query=query)})

        chat_completion = self.model.chat.completions.create(
                            messages=message,
                            model=self.path,
                            max_completion_tokens=MAX_TOKENS,
                            stream=False)

        response = chat_completion.choices[0].message.content

        return response



