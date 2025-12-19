from abc import ABC, abstractmethod
from openai import OpenAI

class ModelAdaptor(ABC):
    client : OpenAI
    model : str

    def __init__(self, **kwargs):
        """
        Initialize the ModelAdaptor with a client and model name.
        :param kwargs: Optional parameters for client and model name.
        """
        self.client = kwargs.get('client')
        self.model = kwargs.get('model')

        # If no client is provided, use the default OpenAI client
        if not self.client:
            client = OpenAI(api_key="ollama" , base_url="http://localhost:11434/v1")
            self.client = client
            print("No client provided.Use ollama client...")
        # If no model name is provided, raise an error
        if not self.model:
            ValueError("Model name is required.")

    def generate(self, **kwargs) -> str:

        # 提示词构建
        instruction = kwargs.get('instruction',"")
        # 任务文本
        text = kwargs.get('text', "")
        temperature = kwargs.get('temperature', 0.8)
        
        response = self.client.chat.completions.create(
            model = self.model,
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": text},
            ],
            stream=False,
            temperature=temperature,
        )
        # 获取返回的文本
        responseText = response.choices[0].message.content
        return responseText
        

