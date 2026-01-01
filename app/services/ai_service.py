import torch
import uuid
from typing import List
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def generate_answer(query: str, context: str, tokenizer: GPT2Tokenizer, llm_model: GPT2LMHeadModel) -> str:
    truncated_context = context[:700]
    prompt = f"Context: {truncated_context}\n\nQuestion: {query}\n\nAnswer:"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    
    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=100,
            no_repeat_ngram_size=2,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full_response.split("Answer:")[-1].strip()