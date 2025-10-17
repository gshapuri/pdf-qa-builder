from __future__ import annotations

import os
import warnings
from typing import List, Dict, Any

from tqdm import tqdm

from pdf_pipeline.text_splitter import Chunk


# predefined model and their token limits
DEFAULT_MODELS = {
    "fast": {"model": "Qwen/Qwen2.5-0.5B-Instruct", "max_tokens": 32768, "type": "slm"},
    "balanced": {"model": "Qwen/Qwen2.5-1.5B-Instruct", "max_tokens": 32768, "type": "slm"},
    "openai": {"model": "gpt-4o-mini", "max_tokens": 128000, "type": "api"},
}


class Summarizer:
    """
    generate short summaries for text sections or chunks
    
    supports three backend types:
    - fast: qwen 0.5b (fastest, good quality, runs in codespaces)
    - balanced: qwen 1.5b (better quality, still fast)
    - openai: gpt-4o-mini (best quality, requires api key)
    """

    def __init__(self, model_name: str = "fast", max_length: int = 120):
        """
        initialize summarizer with automatic backend selection
        
        :param model_name: model preset ('fast', 'balanced', 'openai') or full model path
        :param max_length: maximum summary length in tokens
        """
        # resolve preset names to actual model paths
        if model_name in DEFAULT_MODELS:
            model_config = DEFAULT_MODELS[model_name]
            resolved_model = model_config["model"]
            self.max_input_tokens = model_config["max_tokens"]
            self.model_type = model_config["type"]
        else:
            resolved_model = model_name
            self.max_input_tokens = 4096
            self.model_type = "slm"
        
        self.model_name = resolved_model
        self.max_length = max_length
        
        # detect available backend
        has_openai_key = bool(os.getenv("OPENAI_API_KEY"))
        self.use_openai = (has_openai_key and self.model_type == "api") or resolved_model.startswith("gpt-")
        
        if self.use_openai:
            if not has_openai_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            from openai import OpenAI
            self.client = OpenAI()
            print(f"Using openai {resolved_model} backend")
        else:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            print(f"Loading small language model: {resolved_model}")
            self.tokenizer = AutoTokenizer.from_pretrained(resolved_model)
            self.model = AutoModelForCausalLM.from_pretrained(
                resolved_model,
                low_cpu_mem_usage=True,
            )
            self.model.eval()  # set to evaluation mode
            print(f"Model loaded, max input tokens: {self.max_input_tokens}")

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        truncate text to approximately max_tokens
        
        :param text: input text
        :param max_tokens: maximum number of tokens
        :return: truncated text
        """
        max_words = int(max_tokens * 0.75)
        
        words = text.split()
        if len(words) <= max_words:
            return text
        
        truncated_words = words[:max_words]
        return " ".join(truncated_words)

    def summarize_text(self, text: str) -> str:
        """
        generate a concise summary of input text
        
        :param text: input text to summarize
        :return: short summary string
        """
        if not text or len(text.strip()) == 0:
            return ""
        
        if self.use_openai:
            return self._summarize_with_openai(text)
        
        return self._summarize_with_slm(text)

    def _summarize_with_openai(self, text: str) -> str:
        """
        summarize text using openai model
        
        :param text: input text
        :return: summary string
        """
        truncated = self._truncate_to_tokens(text, self.max_input_tokens - 100)
        
        prompt = (
            "summarize the following text clearly and factually in 3-5 sentences:\n\n"
            f"{truncated}"
        )
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "you are a helpful scientific summarizer"},
                {"role": "user", "content": prompt},
            ],
            max_tokens=250,
        )
        
        return response.choices[0].message.content.strip()

    def _summarize_with_slm(self, text: str) -> str:
        """
        summarize text using small language model
        
        :param text: input text
        :return: summary string
        """
        import torch
        
        safe_limit = int(self.max_input_tokens * 0.5)
        truncated = self._truncate_to_tokens(text, safe_limit)
        
        # create chat prompt
        messages = [
            {"role": "system", "content": "you are a helpful assistant that summarizes scientific text concisely"},
            {"role": "user", "content": f"summarize the following text in 3-5 sentences:\n\n{truncated}"}
        ]
        
        try:
            # apply chat template
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
            
            # generate
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                with torch.no_grad():  # disable gradient calculation
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=150,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
            
            # decode only the generated part
            generated_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            return generated_text.strip()
            
        except Exception as e:
            print(f"Warning: slm summarization failed, using fallback: {e}")
            sentences = truncated.split(". ")
            return ". ".join(sentences[:3]) + "."

    def summarize_sections(self, sections: Dict[str, str]) -> Dict[str, str]:
        """
        summarize a mapping of section names to text
        
        :param sections: dict of section names and text
        :return: dict of section names and summaries
        """
        summaries: Dict[str, str] = {}
        for name, text in tqdm(sections.items(), desc="summarizing sections"):
            summaries[name] = self.summarize_text(text)
        
        return summaries

    def summarize_chunks(self, chunks: List[Chunk]) -> List[Dict[str, Any]]:
        """
        summarize a list of chunk objects
        
        :param chunks: list of chunk objects
        :return: list of dicts with section, chunk_id, text, summary
        """
        summarized: List[Dict[str, Any]] = []
        for chunk in tqdm(chunks, desc="summarizing chunks"):
            summary = self.summarize_text(chunk.text)
            summarized.append({
                "section": chunk.section,
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "summary": summary,
            })
        
        return summarized
