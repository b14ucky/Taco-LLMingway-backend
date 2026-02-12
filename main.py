from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from taco_llmingway import GPT
from taco_llmingway.tokenizer import Tokenizer
import torch
from pathlib import Path
from pydantic import BaseModel

app = FastAPI()

tokenizer_path = Path("model/tokenizer.json")
config_path = Path("model/config.json")
weights_path = Path("model/taco-llmingway.pth")

tokenizer = Tokenizer.load(path=tokenizer_path)
model = GPT.load(config_path, weights_path, device="cpu")


class GenerateQuery(BaseModel):
    starting_text: str
    n_iters: int
    temperature: float = 1.0


@app.post("/generate")
async def generate(query: GenerateQuery):
    tokens = torch.tensor(
        tokenizer.encode(query.starting_text), dtype=torch.long
    ).unsqueeze(0)

    new_tokens = model.generate(
        tokens, n_iters=query.n_iters, temperature=query.temperature
    )

    return {"generated_text": tokenizer.decode(new_tokens.squeeze().tolist())}


@app.post("/stream")
async def stream(query: GenerateQuery):
    tokens = torch.tensor(
        tokenizer.encode(query.starting_text), dtype=torch.long
    ).unsqueeze(0)

    stream_gen = model.stream(
        tokens,
        n_iters=query.n_iters,
        tokenizer=tokenizer,
        temperature=query.temperature,
    )

    return StreamingResponse(stream_gen, media_type="text/plain")
