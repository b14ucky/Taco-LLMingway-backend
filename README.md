# Taco LLMingway - Backend

The inference engine for the Taco LLMingway project. This is a FastAPI-based server that loads a custom GPT model trained on Taco Hemingway's lyrics and serves predictions via standard JSON or high-performance streaming.

## Features

- **Real-time Streaming**: Implements `StreamingResponse` to deliver lyrics token-by-token.
- **Dual Endpoints**: Supports both standard POST requests and streaming for interactive UIs.
- **PyTorch Integration**: Optimized for CPU inference using pre-trained weights.
- **CORS Enabled**: Configured for seamless communication with the Next.js frontend.

## Trained Model

The model weights used in this project were trained on a custom dataset of Taco Hemingway's lyrics. You can find the pre-trained weights and tokenizer on Kaggle:

**[Taco LLMingway on Kaggle](https://www.kaggle.com/models/b14ucky/taco-llmingway)**

## Tech Stack

- **Framework**: FastAPI
- **Inference**: PyTorch
- **Server**: Uvicorn (Dev) / Gunicorn (Prod)
- **Data Validation**: Pydantic

## Project Structure

```bash
├── main.py              # Application entry point & routes
├── requirements.txt     # Python dependencies
└── model/               # Download weights from Kaggle here
    ├── tokenizer.json
    ├── config.json
    └── taco-llmingway.pth
```

## Installation & Setup

1.  **Clone the repositories**:
    Ensure both the backend and the core library are in the same parent directory:

    ```bash
    git clone https://github.com/your-username/taco-llmingway-backend.git
    git clone https://github.com/your-username/taco-llmingway.git
    ```

2.  **Install the core library**:

    ```bash
    pip install ../taco-llmingway
    cd taco-llmingway-backend
    ```

3.  **Install remaining dependencies**:
    For CPU-only VPS:
    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install -r requirements.txt
    ```

## Running the App

### Development (Uvicorn)

Use this for local testing and development with hot-reload:

```bash
uvicorn main:app --reload --port 8000
```

### Production (Gunicorn)

Use Gunicorn with the Uvicorn worker class for better stability and performance:

```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000
```

## API Reference

**POST** `/stream`

```json
{
	"starting_text": "Wracam ekspresem do Warszawy...",
	"n_iters": 256,
	"temperature": 0.8
}
```

**POST** `/generate`

```json
{
	"starting_text": "Wracam ekspresem do Warszawy...",
	"n_iters": 256,
	"temperature": 0.8
}
```

\*_Note: The `/stream` endpoint will return a streaming response, while `/generate` will return the raw tokens._

## Related Repositories

- [Taco-LLMingway-frontend](https://github.com/b14ucky/taco-llmingway-frontend) - The Next.js web interface.
- [Taco-LLMingway](https://github.com/b14ucky/taco-llmingway) - Training scripts and data processing.

## License

This project is licensed under the **MIT License**.
