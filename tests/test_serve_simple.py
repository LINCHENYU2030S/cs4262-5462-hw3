from typing import List, Tuple

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import serve

DEFAULT_MODEL = "gpt2"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MockSeqState:
    def __init__(self, prompt: str, request_id: str, embedding_only: bool = False):
        self.input_ids: torch.Tensor
        self.past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...] = ()
        self.decoded_tokens = ""
        self.has_prefilled = False
        self.generated_tokens = 0
        self.prompt = prompt
        self.status = "pending"
        self.request_id = request_id
        self.embedding_only = embedding_only
        self.embedding: List[float] = []


@pytest.fixture(scope="module")
def tokenizer_and_model():
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL).to(DEFAULT_DEVICE)  # type: ignore
    return tokenizer, model


def test_retrieve_past_key_values():
    batch_size = 2
    num_layers = 2
    num_heads = 4
    head_dim = 8
    padded_seq_len = 10
    unpadded_seq_len = 5

    # Dummy past key values
    past_key_values = tuple(
        (
            torch.randn(batch_size, num_heads, padded_seq_len, head_dim),  # keys
            torch.randn(batch_size, num_heads, padded_seq_len, head_dim),  # values
        )
        for _ in range(num_layers)
    )

    new_past_key_values = serve.retrieve_past_key_values(
        past_key_values, 0, unpadded_seq_len
    )

    # Check past key-values shape
    assert len(new_past_key_values) == num_layers
    for keys, values in new_past_key_values:
        assert keys.shape == (num_heads, unpadded_seq_len, head_dim)
        assert values.shape == (num_heads, unpadded_seq_len, head_dim)


def test_prepare_inputs_for_prefill(tokenizer_and_model):
    tokenizer, model = tokenizer_and_model
    seq_states = [
        MockSeqState(prompt="Hello, how are you?", request_id="1"),
        MockSeqState(prompt="What is your name?", request_id="2"),
    ]

    input_ids = [
        tokenizer(seq_state.prompt, return_tensors="pt").input_ids[0]
        for seq_state in seq_states
    ]
    padded_input_ids, attention_mask = serve.prepare_inputs_for_prefill(
        seq_states, tokenizer, model  # type: ignore
    )

    # Check shapes
    num_seqs = len(seq_states)
    max_len = max(len(input_id) for input_id in input_ids)
    assert padded_input_ids.shape == (num_seqs, max_len)
    assert attention_mask.shape == (num_seqs, max_len)


def test_prefill(tokenizer_and_model):
    tokenizer, model = tokenizer_and_model
    n_layer = model.config.n_layer
    n_head = model.config.n_head
    head_dim = model.config.n_embd // model.config.n_head

    seq_states = [
        MockSeqState(prompt="Hello, how are you?", request_id="1"),
        MockSeqState(prompt="What is your name?", request_id="2"),
    ]

    seq_states = serve.prefill(seq_states, model, tokenizer)  # type: ignore

    # Check past key-values shape
    input_ids_list = [
        tokenizer(seq_state.prompt, return_tensors="pt").input_ids[0]
        for seq_state in seq_states
    ]
    for input_ids, seq_state in zip(input_ids_list, seq_states):
        assert seq_state.has_prefilled
        assert len(seq_state.past_key_values) == n_layer
        keys, values = seq_state.past_key_values[0]
        kv_shape = (n_head, len(input_ids), head_dim)
        assert keys.shape == kv_shape
        assert values.shape == kv_shape


def test_prepare_inputs_for_decode(tokenizer_and_model):
    tokenizer, model = tokenizer_and_model
    n_layer = model.config.n_layer
    n_head = model.config.n_head
    head_dim = model.config.n_embd // model.config.n_head

    seq_states = [
        MockSeqState(prompt="Hello, how are you?", request_id="1"),
        MockSeqState(prompt="What is your name?", request_id="2"),
    ]

    seq_states = serve.prefill(seq_states, model, tokenizer)  # type: ignore

    attention_mask, past_key_values, _ = serve.prepare_inputs_for_decode(seq_states)

    # Check attention mask shape
    input_ids_list = [
        tokenizer(seq_state.prompt, return_tensors="pt").input_ids[0]
        for seq_state in seq_states
    ]
    num_seqs = len(seq_states)
    max_len = max(len(input_ids) for input_ids in input_ids_list)
    assert attention_mask.shape == (num_seqs, max_len + 1)

    # Check past key-values shape
    past_kv_tuples = past_key_values.to_legacy_cache()
    kv_shape = (num_seqs, n_head, max_len, head_dim)
    assert len(past_kv_tuples) == n_layer
    for keys, values in past_kv_tuples:
        assert keys.shape == kv_shape
        assert values.shape == kv_shape


def test_decode(tokenizer_and_model):
    tokenizer, model = tokenizer_and_model
    n_layer = model.config.n_layer
    n_head = model.config.n_head
    head_dim = model.config.n_embd // model.config.n_head

    seq_states = [
        MockSeqState(prompt="Hello, how are you?", request_id="1"),
        MockSeqState(prompt="What is your name?", request_id="2"),
    ]

    seq_states = serve.prefill(seq_states, model, tokenizer)  # type: ignore
    seq_states = serve.decode(seq_states, model, tokenizer)

    # Check past key-values shape
    input_ids_list = [
        tokenizer(seq_state.prompt, return_tensors="pt").input_ids[0]
        for seq_state in seq_states
    ]
    for input_ids, seq_state in zip(input_ids_list, seq_states):
        assert seq_state.has_prefilled
        assert len(seq_state.past_key_values) == n_layer
        keys, values = seq_state.past_key_values[0]
        kv_shape = (n_head, len(input_ids) + 1, head_dim)
        assert keys.shape == kv_shape
        assert values.shape == kv_shape


def test_embedding_only(tokenizer_and_model):
    tokenizer, model = tokenizer_and_model
    n_embd = model.config.n_embd

    seq_states = [
        MockSeqState(prompt="Hello, how are you?", request_id="1", embedding_only=True),
        MockSeqState(prompt="What is your name?", request_id="2", embedding_only=True),
    ]

    seq_states = serve.embedding_only(seq_states, model, tokenizer)  # type: ignore

    # Check embedding type and shape
    for seq_state in seq_states:
        assert seq_state.embedding_only
        assert isinstance(seq_state.embedding, list)
        assert len(seq_state.embedding) == n_embd
