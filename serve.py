from typing import Tuple, List, TYPE_CHECKING

import torch
from transformers import DynamicCache

if TYPE_CHECKING:
    from api import SeqState


def retrieve_past_key_values(
    past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    i: int,
    generated_tokens: int,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    # retrieve the past key values
    new_past_key_values = []

    # TODO: retrieve the past key values
    # the kv cache is like:
    # past_key_values = (
    #     # layer 1
    #     (keys, values),
    #     # layer 2
    #     (keys, values),
    #     ...
    # )
    # and the size of key/value tensor is (batch_size, num_heads, padded_seq_len, head_dim)
    # you need to retrieve the past key values for the every sequence in the batch
    # the size of the retrieved key/value should be (num_heads, unpadded_seq_len, head_dim) with same structure
    # ==== start your code here ====
    for keys, values in past_key_values:
        new_past_key_values.append(
            (
                keys[i, :, :generated_tokens, :].clone(),
                values[i, :, :generated_tokens, :].clone(),
            )
        )
    # ==== end of your code ====
    return tuple(new_past_key_values)


def prepare_inputs_for_prefill(
    seq_states: List["SeqState"], tokenizer, model
) -> Tuple[torch.Tensor, torch.Tensor]:
    # generate the input ids
    for seq_state in seq_states:
        seq_state.input_ids = (
            tokenizer(seq_state.prompt, return_tensors="pt")
            .input_ids[0]
            .to(model.device)
        )
    input_ids = [seq_state.input_ids for seq_state in seq_states]

    # TODO: pad the input ids
    # pad the input ids with the eos_token to the max length in the batch
    # ==== start your code here ====
    padded_input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=tokenizer.eos_token_id,
    )
    # ==== end of your code ====


    # TODO: pad the attention mask
    # pad the attention mask with 0 to the max length in the batch
    # ==== start your code here ====
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [torch.ones_like(seq_input_ids, dtype=torch.long) for seq_input_ids in input_ids],
        batch_first=True,
        padding_value=0,
    )
    # ==== end of your code ====
    return padded_input_ids, attention_mask


def prepare_inputs_for_decode(
    seq_states: List["SeqState"],
) -> Tuple[torch.Tensor, DynamicCache, torch.Tensor]:
    # cat the input ids
    input_ids = torch.stack([seq_state.input_ids for seq_state in seq_states], dim=0)

    # TODO: pad the attention mask
    # pad the attention mask with 0 to the max length in the batch
    # attention mask shape: (batch_size, seq_len)
    # ==== start your code here ====
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [
            torch.ones(
                seq_state.past_key_values[0][0].shape[1] + seq_state.input_ids.shape[0],
                dtype=torch.long,
                device=seq_state.input_ids.device,
            )
            for seq_state in seq_states
        ],
        batch_first=True,
        padding_value=0,
    )
    # ==== end of your code ====


    # TODO: pad past key values
    # pad the past key values with 0 to the max length in the batch
    # ==== start your code here ====
    padded_past_key_values = []
    for layer_idx in range(len(seq_states[0].past_key_values)):
        layer_keys = [
            seq_state.past_key_values[layer_idx][0].permute(1, 0, 2)
            for seq_state in seq_states
        ]
        layer_values = [
            seq_state.past_key_values[layer_idx][1].permute(1, 0, 2)
            for seq_state in seq_states
        ]
        padded_keys = torch.nn.utils.rnn.pad_sequence(
            layer_keys,
            batch_first=True,
            padding_value=0,
        ).permute(0, 2, 1, 3)
        padded_values = torch.nn.utils.rnn.pad_sequence(
            layer_values,
            batch_first=True,
            padding_value=0,
        ).permute(0, 2, 1, 3)
        padded_past_key_values.append((padded_keys, padded_values))
    # ==== end of your code ====
    return (
        attention_mask,
        DynamicCache.from_legacy_cache(tuple(padded_past_key_values)),
        input_ids,
    )


def embedding_only(seq_states: List["SeqState"], model, tokenizer) -> List["SeqState"]:
    padded_input_ids, attention_mask = prepare_inputs_for_prefill(
        seq_states, tokenizer, model
    )

    # forward
    out = model.forward(
        padded_input_ids, attention_mask=attention_mask, output_hidden_states=True
    )

    # TODO: get the embedding
    # you need to get the embedding of the last layer's output (mean across sequence)
    # and set it to the embedding attribute of each seq_state
    # ==== start your code here ====
    last_hidden_states = out.hidden_states[-1]
    mask = attention_mask.unsqueeze(-1).to(last_hidden_states.dtype)
    embeddings = (last_hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    for seq_state, embedding in zip(seq_states, embeddings):
        seq_state.embedding = embedding.detach().cpu().float().tolist()
    # ==== end of your code ====
    return seq_states


def prefill(seq_states: List["SeqState"], model, tokenizer) -> List["SeqState"]:
    padded_input_ids, attention_mask = prepare_inputs_for_prefill(
        seq_states, tokenizer, model
    )
    # forward
    out = model.forward(
        padded_input_ids,
        attention_mask=attention_mask,
        cache_implementation="dynamic",
    )
    # get the next input tokens
    assert isinstance(out.past_key_values, DynamicCache)
    past_key_values = out.past_key_values.to_legacy_cache()
    model_inputs = out.logits[:, -1].argmax(dim=-1).unsqueeze(1)
    decoded_tokens = tokenizer.batch_decode(model_inputs, skip_special_tokens=True)

    # TODO:update the seq states
    # including: generated_tokens(int), input_ids(torch.Tensor), has_prefilled(bool), decoded_tokens(str), past_key_values(tuple[tuple[torch.Tensor, torch.Tensor], ...])
    # ==== start your code here ====
    prompt_lengths = [seq_state.input_ids.shape[0] for seq_state in seq_states]
    for i, seq_state in enumerate(seq_states):
        seq_state.generated_tokens = prompt_lengths[i] + model_inputs[i].shape[0]
        seq_state.input_ids = model_inputs[i]
        seq_state.has_prefilled = True
        seq_state.decoded_tokens = decoded_tokens[i]
        seq_state.past_key_values = retrieve_past_key_values(
            past_key_values, i, prompt_lengths[i]
        )
    # ==== end of your code ====
    return seq_states


def decode(seq_states: List["SeqState"], model, tokenizer) -> List["SeqState"]:
    # extend the attention mask
    attention_mask, past_key_values, input_ids = prepare_inputs_for_decode(seq_states)
    # forward
    out = model.forward(
        input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        cache_implementation="dynamic",
    )
    # get the next input token
    assert isinstance(out.past_key_values, DynamicCache)
    past_key_values = out.past_key_values.to_legacy_cache()
    model_inputs = out.logits[:, -1].argmax(dim=-1).unsqueeze(1)
    decoded_tokens = tokenizer.batch_decode(model_inputs, skip_special_tokens=True)
    # update the seq states
    # TODO:update the seq states
    # including: generated_tokens(int), input_ids(torch.Tensor), decoded_tokens(str), past_key_values(tuple[tuple[torch.Tensor, torch.Tensor], ...])
    # ==== start your code here ====
    cached_lengths = [seq_state.past_key_values[0][0].shape[1] for seq_state in seq_states]
    input_lengths = [seq_state.input_ids.shape[0] for seq_state in seq_states]
    for i, seq_state in enumerate(seq_states):
        seq_state.generated_tokens += model_inputs[i].shape[0]
        seq_state.input_ids = model_inputs[i]
        seq_state.decoded_tokens += decoded_tokens[i]
        seq_state.past_key_values = retrieve_past_key_values(
            past_key_values,
            i,
            cached_lengths[i] + input_lengths[i],
        )
    # ==== end of your code ====
    return seq_states


def serve_step(model, tokenizer, seq_states: List["SeqState"]) -> int:
    # prefill if any sequence not prefilled
    prefill_list = []
    embedding_only_list = []
    for seq_state in seq_states:
        if not seq_state.has_prefilled:
            prefill_list.append(seq_state)
        if seq_state.embedding_only:
            embedding_only_list.append(seq_state)

    if len(embedding_only_list) != 0:
        embedding_only(embedding_only_list, model, tokenizer)
        consumed_tokens = 0

    elif len(prefill_list) != 0:
        prefill(prefill_list, model, tokenizer)
        consumed_tokens = sum(seq_state.generated_tokens for seq_state in prefill_list)

    else:
        decode(seq_states, model, tokenizer)
        consumed_tokens = len(seq_states)

    return consumed_tokens
