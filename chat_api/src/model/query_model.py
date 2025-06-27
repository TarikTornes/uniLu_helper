from groq import Groq


class QueryModel():

    def __init__(self, path, api_key):
        self.model = Groq(api_key=api_key)
        self.path = path


    def opt_query(self, query, chat_history=None, MAX_TOKENS=128):

        message = [{
                "role": "system",
                "content": """You are a query optimization assistant. Your task is to refine the user’s input into a concise, unambiguous, and context-rich query optimized for similarity search in a RAG-based chatbot.
                ## Instructions:
                - Condense: Remove filler words while retaining all key context from the conversation history.
                - Clarify: Resolve pronouns, vague terms, or ambiguous references using chat history.
                - Enrich: Add implicit context (e.g., timeframes, entities, intent) if missing but critical.
                - Focus: Eliminate redundancy. Prioritize information directly relevant to retrieval.
                - Structure: Output only the optimized query. No explanations or markdown.

                Return ONLY the optimized query. Do NOT include extra text."""}]

        if chat_history:
            message.extend(chat_history)

        message.append({"role": "user",
                        "content":
                        f"""Improve the user query using the following chat history if they are relevant.
                        
                        ## Query: {query}"""})

        chat_completion = self.model.chat.completions.create(
                            messages=message,
                            model=self.path,
                            max_completion_tokens=MAX_TOKENS,
                            stream=False)

        response = chat_completion.choices[0].message.content

        return [response]



