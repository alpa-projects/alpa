import argparse
from typing import Dict, Optional, Union, Sequence

import requests

DEFAULT_URL = "https://api.alpa.ai"

headers = {"User-Agent": "Alpa Client"}


class Client(object):

    def __init__(self,
                 url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 default_model: str = "default") -> None:
        if url is None:
            url = DEFAULT_URL

        self.api_key = api_key
        self.default_model = default_model
        self.completions_url = url + "/completions"
        self.logprobs_url = url + "/logprobs"

    def completions(
        self,
        prompt: Union[str, Sequence[str], Sequence[int], Sequence[Sequence[int]]],
        min_tokens: int = 0,
        max_tokens: int = 32,
        top_p: float = 1.0,
        temperature: float = 1.0,
        echo: bool = True,
        model: Optional[str] = None,
    ) -> Dict:
        """
        Generation API.
        Parameters match those of the OpenAI API.
        https://beta.openai.com/docs/api-reference/completions/create

        Args:
          prompt: a list of tokenized inputs.
          min_tokens: The minimum number of tokens to generate.
          max_tokens: The maximum number of tokens to generate.
          temperature: What sampling temperature to use.
          top_p: The nucleus sampling probability.
          echo: if true, returned text/tokens/scores includes the prompt.
        """
        pload = {
            "model": model or self.default_model,
            "prompt": prompt,
            "min_tokens": min_tokens,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "echo": echo,
            "api_key": self.api_key
        }
        result = requests.post(self.completions_url, json=pload, headers=headers)
        return self.result_or_error(result)

    def logprobs(
        self,
        prompt: Union[str, Sequence[str], Sequence[int], Sequence[Sequence[int]]],
        top_k: int = 50,
        cache_id: Optional = None,
        model: Optional[str] = None) -> Dict:
        """Return the log probability of the next top-k tokens"""
        pload = {
            "model": model or self.default_model,
            "prompt": prompt,
            "top_k": top_k,
            "api_key": self.api_key
        }
        if cache_id:
            pload["cache_id"] = cache_id
        result = requests.post(self.logprobs_url, json=pload, headers=headers)
        return self.result_or_error(result)

    def result_or_error(self, result):
        result = result.json()
        if result.get("type", "") == "error":
            raise RuntimeError(
                result["stacktrace"] +
                f'RuntimeError("{result["message"]}")')
        else:
            return result
