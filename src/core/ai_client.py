import os
from typing import List, Dict, Optional, Tuple
from groq import Groq


class GroqAI:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment.")
        self.client = Groq(api_key=api_key)












    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.2,
        top_p: float = 1.0,
        max_tokens: int = 256,
        stop: Optional[list] = None,
    ) -> Tuple[str, dict, Optional[str]]:
        






        """
        Returns (text, usage_dict, error_str)
        """







        try:
            resp = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=stop,
            )
            text = resp.choices[0].message.content or ""
            usage = getattr(resp, "usage", {}) or {}
            return text.strip(), usage, None
        except Exception as e:
            return "", {}, f"❌ Groq API Error: {e}"
