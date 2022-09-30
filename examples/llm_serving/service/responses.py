"""Adapted from Metaseq."""
import uuid
import time


class OAIResponse:

    def __init__(self, results: list) -> None:
        self.results = results
        self.response_id = str(uuid.uuid4())
        self.created = int(time.time())

    def __dict__(self):
        return {
            "id":
                self.response_id,
            "object":
                "text_completion",
            "created":
                self.created,
            "choices": [
                {
                    "text": result["text"],
                    # TODO(Hao): align with what OpenAI returns
                    # "logprobs": {
                    #     "tokens": result["tokens"],
                    #     "token_logprobs": result["token_scores"],
                    #     "text_offset": result["text_offset"],
                    #     "top_logprobs": result["top_logprobs"],
                    #     "finish_reason": "length",
                    # },
                } for result in self.results
            ],
        }
