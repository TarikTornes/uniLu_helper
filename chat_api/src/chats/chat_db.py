import redis, json

class ChatDB():
    def __init__(self, host, passwd, port, dec_resp, expiry):

        self.client = redis.Redis(host=host, port=port, password=passwd, decode_responses= dec_resp)
        self.session_ttl = expiry


    def add_message(self, session_key, content, role):

        message = json.dumps({"role": role, "content": content})
        self.client.rpush(session_key, message)  # Append message to list
        self.client.expire(session_key, self.session_ttl)  # Reset TTL to 20 minutes


    def get_history(self, session_key):
        chats = self.client.lrange(session_key, -10, -1)
        return [json.loads(msg) for msg in chats]

    def kill_session_history(self, session_key: int):
        """
        Deletes all chat history associated with the given session_key.
        """
        self.client.delete(session_key)
