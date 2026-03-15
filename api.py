import argparse
import asyncio
import sys
import uuid
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Tuple, Union

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

from serve import serve_step


class Request(BaseModel):
    prompt: str


class RequestStatus:
    PENDING = "pending"
    COMPLETED = "completed"
    ERROR = "error"
    QUOTA_EXCEEDED = "quota_exceeded"


class GenerateResponse(BaseModel):
    text: Optional[str]
    status: str


class EmbeddingResponse(BaseModel):
    embedding: List[float]


class SeqState:
    def __init__(self, prompt: str, request_id: str, embedding_only: bool = False):
        self.input_ids: torch.Tensor
        self.past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...] = ()
        self.decoded_tokens = ""
        self.has_prefilled = False
        self.generated_tokens = 0
        self.prompt = prompt
        self.status = RequestStatus.PENDING
        self.request_id = request_id
        self.embedding_only = embedding_only
        self.embedding: List[float] = []


class RequestPool:
    def __init__(self, init_quota: int = 1000, max_generated_tokens: int = 50):
        self.requests: Dict[str, SeqState] = {}
        self.active_requests: Dict[str, SeqState] = {}
        self.max_active_requests = 2
        self.quota = init_quota
        self.max_generated_tokens = max_generated_tokens
        self.queue = asyncio.Queue()
        self.lock = asyncio.Lock()

    async def add_request(self, request: Request, embedding_only: bool = False) -> str:
        request_id = str(uuid.uuid4())
        request_data = SeqState(request.prompt, request_id, embedding_only)

        async with self.lock:
            self.requests[request_id] = request_data
            await self.queue.put(request_id)

        return request_id

    async def wait_for_completion(
        self, request_id: str, interval: float = 0.1
    ) -> Union[EmbeddingResponse, GenerateResponse]:
        """wait for request completion"""
        while True:
            # TODO: wait for completion
            # wait until the request is completed and return the response
            # for embedding, you should return EmbeddingResponse(embedding=seq.embedding)
            # for generate, you should return GenerateResponse(text=seq.decoded_tokens, status=seq.status)
            # ==== start your code here ====
            async with self.lock:
                seq = self.requests.get(request_id)
                if seq is not None and seq.status != RequestStatus.PENDING:
                    self.requests.pop(request_id, None)
                    if seq.embedding_only:
                        return EmbeddingResponse(embedding=seq.embedding)
                    return GenerateResponse(text=seq.decoded_tokens, status=seq.status)
            await asyncio.sleep(interval)
            # ==== end of your code ====

    def stop_generation(self, tokenizer) -> List[SeqState]:
        # TODO: stop generation
        # stop generation if:
        # the generated tokens exceed the self.max_generated_tokens
        # or the last token is eos_token
        # or the sequence is an embedding only sequence (seq.embedding_only == True)
        # ==== start your code here ====
        stop_list = []
        for seq in self.active_requests.values():
            last_token_is_eos = (
                hasattr(seq, "input_ids")
                and seq.input_ids.numel() > 0
                and tokenizer.eos_token_id is not None
                and seq.input_ids[-1].item() == tokenizer.eos_token_id
            )
            if (
                seq.generated_tokens >= self.max_generated_tokens
                or last_token_is_eos
                or seq.embedding_only
            ):
                stop_list.append(seq)
        # ==== end of your code ====
        return stop_list

    @torch.no_grad()
    async def process_request(self, model, tokenizer) -> None:
        while True:
            # TODO: get pending requests
            # if active requests are less than max_active_requests,
            # pop requests from the queue (if any) and put it into active requests
            # ==== start your code here ====
            async with self.lock:
                while (
                    len(self.active_requests) < self.max_active_requests
                    and not self.queue.empty()
                ):
                    request_id = self.queue.get_nowait()
                    if request_id in self.requests and request_id not in self.active_requests:
                        self.active_requests[request_id] = self.requests[request_id]
            # ==== end of your code ====

            if self.quota <= 0:
                # TODO: stop all requests if quota is exceeded
                # pop it from the active requests and requests
                # also set the status of the requests to RequestStatus.QUOTA_EXCEEDED
                # ==== start your code here ====
                async with self.lock:
                    for seq_state in self.requests.values():
                        if seq_state.status == RequestStatus.PENDING:
                            seq_state.status = RequestStatus.QUOTA_EXCEEDED
                    self.active_requests.clear()
                    while not self.queue.empty():
                        self.queue.get_nowait()
                # ==== end of your code ====

            if len(self.active_requests) > 0:
                # serve step
                request_data = list(self.active_requests.values())
                consumed_tokens = serve_step(model, tokenizer, request_data)
                # stop generation
                stop_list = self.stop_generation(tokenizer)
                for seq_state in stop_list:
                    seq_state.status = RequestStatus.COMPLETED
                # update quota
                self.quota -= consumed_tokens
                # clean up completed requests
                for req in request_data:
                    if req.status == RequestStatus.COMPLETED:
                        self.active_requests.pop(req.request_id)

            await asyncio.sleep(0.01)  # avoid high CPU usage


def parse_arguments():
    parser = argparse.ArgumentParser(description='API server with configurable init quota')
    parser.add_argument('--port',
                        type=int,
                        default=8000,
                        help='Port to run the API server on (default: 8000)')
    parser.add_argument('--init-quota',
                       type=int, 
                       default=sys.maxsize,
                       help='Initial quota value (default: infinite)')
    parser.add_argument('--max-generated-tokens', 
                       type=int, 
                       default=sys.maxsize,
                       help='Max generated tokens (default: infinite)')
    return parser.parse_args()


# Get command line arguments
args = parse_arguments()

# initialize request pool
request_pool = RequestPool(args.init_quota, args.max_generated_tokens)


# Model
# NOTE: To use meta-llama/Llama-3.2-1B-Instruct, you need to do the following steps:
# 1. Create a Hugging Face account if you don't have one.
# 2. Apply for access to the model at https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct. The access is usually approved within several hours.
# 3. Create a HF access token (see https://huggingface.co/docs/hub/en/security-tokens#user-access-tokens).
# 4. Log in to Hugging Face Hub using the HF CLI (see https://huggingface.co/docs/huggingface_hub/en/quick-start#login-command).
# Otherwise, you may use Qwen/Qwen2.5-1.5B-Instruct, which does not require access permission.
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

if torch.cuda.is_available():
    model_device = "cuda"
    model_dtype = torch.bfloat16
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    model_device = "mps"
    model_dtype = torch.float16
else:
    model_device = "cpu"
    model_dtype = torch.float32

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=model_dtype,
    device_map="auto",
    low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    asyncio.create_task(request_pool.process_request(model, tokenizer))
    yield


app = FastAPI(lifespan=lifespan)


# API endpoints
@app.post("/generate", response_model=GenerateResponse)
async def generate(request: Request):
    request_id = await request_pool.add_request(request)
    return await request_pool.wait_for_completion(request_id)


@app.post("/get_embedding", response_model=EmbeddingResponse)
async def get_embedding(request: Request):
    request_id = await request_pool.add_request(request, embedding_only=True)
    return await request_pool.wait_for_completion(request_id)


if __name__ == "__main__":
    uvicorn.run("api:app", port=args.port, reload=False)
