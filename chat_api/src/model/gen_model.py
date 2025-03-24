from groq import Groq



class Gen_Model:

    def __init__(self, path, api_key):
        print(type(api_key))
        self.model = Groq(api_key=api_key)
        self.path = path


    def get_formatted_prompt(self, instruction, input=None):
        message = [{
            "role": "system",
            "content": 
                    """You are a knowledgeable assistant helping users find accurate information from the website of the University of Luxembourg. 
                    # Your role is to:
                    1. Provide a clear, direct answer based on the provided documents
                    2. Ensure responses are factual and grounded in the source material
                    3. Only respond to questions related to the University
                    4. Maintain a professional and helpful tone

                    # Guidelines:
                    - Focus on the most relevant information from the Input documents
                    - Include ALWAYS exactly one URL for the primary source
                    - Keep responses concise and to the point
                    - If the input or documents is insufficient, unclear or not relevant for the instruction, use the error response template

                    ## Response format for complete and unbiased information:
                    [Concise answer addressing the instruction]
                    Learn more: [URL]
                    ## Response format for unclear or insufficient information:
                    Sorry, I don't have enough information to fully answer your question. You may find relevant details here: [URL]

                    """},

                   {
            "role": "user",
            "content": 
                   f""" 
                    Answer the following instruction using only information from the provided documents if they are relevant.

                    ## Input: {input}

                    ## Instruction:\n {instruction}
                   """
                   }]
           
        return message


    def get_response(self, query_results, INSTRUCTION, MAX_TOKENS=1024):

        chat_completion = self.model.chat.completions.create(
                            messages=self.get_formatted_prompt(instruction=INSTRUCTION, input=query_results),
                            model=self.path,
                            max_completion_tokens=MAX_TOKENS,
                            stream=False)
        print(chat_completion.choices[0].message.content)
        return chat_completion.choices[0].message.content

