from groq import Groq
import json

PROMPT_SYSTEM_REWRITE = """You are a query optimization assistant for the University of Luxembourg's RAG chatbot. Your goal is to turn any user question into a set of concise, unambiguous sub-queries optimized for retrieval. Strictly follow these instructions for each query:

1. **RESOLVE**
    - Replace pronouns/vague terms with specific entities from history

2. **Focus**
    Eliminate redundancy. Prioritize information directly relevant to retrieval.

3. **Query Optimization Technique**
    a. Expand:Generate semantically similar queries or paraphrases of the original query to capture different expressions of the same intent.
    b. Decompose: Break down complex queries into simpler sub-queries, each targeting a distinct aspect of the original question.
    c. Abstract: Translate specific queries into broader concepts to aid in generalization and retrieval.

4. **Output Format**
    Return *only* the optimized queries in a JSON object:
    {
        "optimized_queries": [
            "Example sub.query 1!", 
            "Example sub-query 2!", 
            ...
        ]
    }

5. **Few-Shot Examples**
    Input:
    Are there populations larger than those of China or India?"
    Output:
    {
        "optimized_queries": [
            "What is the nation with the largest population?",
            "What is the population of China?",
            "What is the population of India?"
        ]
    }

    Input: 
    Who is Prof. Theobald and on what does he research on?
    Output:
    {
        "optimized_queries": [
            "Who is Prof. Theobald",
            "What does Prof. Theobald research on?",
            "Research projects and work of Prof. Theobald?"
        ]
    }

Return ONLY the JSON-object. Do NOT include extra text.


"""


PROMPT_USER = """Improve the user query using the previous chat history if relevant.
                
## Query: {user_query}

## Instructions:
- Generate at most {amount} optimized queries by following strictly the aforementioned instructions
- Ensure each query is concise, unambiguous, and context-rich
- Ensure that the queries are optimal for retrieval
- Return only the optimized queries in the specified JSON-format"""


class QueryModel():

    def __init__(self, path, api_key):
        self.model = Groq(api_key=api_key)
        self.path = path


    def opt_query(self, query, chat_history=None, MAX_TOKENS=128, MAX_QUERIES=3) -> list[str]:

        message = [{"role": "system",
                    "content": PROMPT_SYSTEM_REWRITE}]

        if chat_history:
            message.extend(chat_history)


        message.append({"role": "user",
                        "content": PROMPT_USER.format(user_query=query, amount=MAX_QUERIES)})

        print(message)

        chat_completion = self.model.chat.completions.create(
                            messages=message,
                            model=self.path,
                            max_completion_tokens=MAX_TOKENS,
                            stream=False)

        response = chat_completion.choices[0].message.content

        print(response)
        data = json.loads(response)

        queries = data["optimized_queries"]


        return queries[:MAX_QUERIES]



