import json
import openai
import logging
from typing import List, Dict
from src.data_models import QAResponse

class OpenAIPDFQuestionAnswering:
    def __init__(self, config, api_key: str, logger=logging.getLogger(__name__)):
        """Initialize with configuration and API key."""
        self.logger = logger
        self.model_name = config.get("DEV", "model_name", fallback="gpt-4o-mini")
        self.client = openai.OpenAI(api_key=api_key)
        self.logger.info("Initialized OpenAIPDFQuestionAnswering instance.")

    def get_answer_from_openai(self, question: str, context: str) -> Dict[str, any]:
        """Send question and context to OpenAI API and parse the response."""
        prompt = f"""
        Based on the following context, please answer the question. If the answer cannot be determined from the context, respond with "NA".
        
        Context:
        {context}
        
        Question: {question}
        
        Provide your response as JSON:
        {{
            "answer": "your answer",
            "confidence_score": float between 0 and 1
        }}
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        result = json.loads(response.choices[0].message.content)
        self.logger.info("Received response from OpenAI API.")
        return result