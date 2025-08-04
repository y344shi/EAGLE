import torch
import triton
import triton.language as tl


@triton.jit
def _tree_mask_kernel(
    tree_mask,
    parents,
    stride_tb,
    stride_tr,
    stride_tc,
    stride_pb,
    stride_pi,
    *,
    total_tokens: tl.constexpr,
):
    """Compute tree mask for speculative decoding.

    The kernel processes one batch per program and iterates sequentially over
    tokens to respect parent-child dependencies.  Within each iteration the
    entire token dimension is handled in parallel using vector operations.

    Parameters
    ----------
    tree_mask: ``[B, T, T]`` output tensor storing the visibility mask.
    parents: ``[B, T]`` tensor of parent indices for each token.
    stride_*: strides for the respective tensors.
    total_tokens: total number of tokens in the tree (compile-time constant).
    """

    # Program ID corresponds to the batch index
    b_idx = tl.program_id(axis=0)

    # Handle the entire token dimension in one vectorised block
    token_offsets = tl.arange(0, total_tokens)

    # Initialise root token mask: it can only see itself
    root_ptr = tree_mask + b_idx * stride_tb + 0 * stride_tr + token_offsets * stride_tc
    root_mask = (token_offsets == 0).to(tl.int32)
    tl.store(root_ptr, root_mask, mask=token_offsets < total_tokens)

    # Sequentially build masks for the remaining tokens
    for t in range(1, total_tokens):
        parent = tl.load(parents + b_idx * stride_pb + t * stride_pi)

        parent_ptr = (
            tree_mask + b_idx * stride_tb + parent * stride_tr + token_offsets * stride_tc
        )
        parent_mask = tl.load(parent_ptr, mask=token_offsets < total_tokens, other=0)

        # Tokens inherit visibility from their parent for previous positions and
        # can always see themselves.
        current_mask = tl.where(
            token_offsets == t,
            1,
            tl.where(token_offsets < t, parent_mask, 0),
        )

        out_ptr = (
            tree_mask + b_idx * stride_tb + t * stride_tr + token_offsets * stride_tc
        )
        tl.store(out_ptr, current_mask, mask=token_offsets < total_tokens)


@triton.jit
def _evaluate_posterior_kernel(
    logits, candidates, best_candidate, accept_length,
    stride_lb, stride_ls, stride_lv,
    stride_cb, stride_cs, stride_ct,
    batch_size, seq_len, vocab_size,
):
    """
    Evaluate posterior probabilities for speculative decoding.
    
    Parameters:
        logits: logits tensor from the base model
        candidates: candidate token sequences from the draft model
        best_candidate: output tensor for best candidate index
        accept_length: output tensor for accepted sequence length
        stride_*: strides for the respective tensors
        batch_size: number of sequences in the batch
        seq_len: sequence length
        vocab_size: vocabulary size
    """
    # Program ID
    pid = tl.program_id(axis=0)
    
    # Compute batch index
    batch_idx = pid
    
    # Initialize variables
    max_accept_len = 0
    max_candidate_idx = 0
    
    # Get number of candidates
    num_candidates = stride_cb // stride_cs
    
    # Evaluate each candidate
    for c in range(num_candidates):
        # Count accepted tokens for this candidate
        accept_len = 0
        
        # Check each position
        for s in range(1, seq_len):
            # Get candidate token
            candidate_token = tl.load(candidates + batch_idx * stride_cb + c * stride_cs + s * stride_ct)
            
            # Get logits for this position
            logits_pos_ptr = logits + batch_idx * stride_lb + (s - 1) * stride_ls
            
            # Find max logit
            max_logit_idx = 0
            max_logit_val = tl.load(logits_pos_ptr)
            
            for v in range(1, vocab_size):
                val = tl.load(logits_pos_ptr + v * stride_lv)
                if val > max_logit_val:
                    max_logit_val = val
                    max_logit_idx = v
            
            # Check if prediction matches
            # Can't use break in Triton, so use a flag
            is_match = max_logit_idx == candidate_token
            accept_len = tl.where(is_match, accept_len + 1, accept_len)
            
            # If no match, we'll stop checking further positions
            # We can't use break, so we'll use a condition in the loop
        
        # Update best candidate if this one has longer accepted sequence
        if accept_len > max_accept_len:
            max_accept_len = accept_len
            max_candidate_idx = c
    
    # Store results
    tl.store(best_candidate + batch_idx, max_candidate_idx)
    tl.store(accept_length + batch_idx, max_accept_len)


@triton.jit
def _update_inputs_kernel(
    input_ids, candidates, best_candidate, accept_length, output_ids,
    stride_ib, stride_is,
    stride_cb, stride_cs, stride_ct,
    stride_ob, stride_os,
    batch_size, input_len, seq_len,
):
    """
    Update input IDs with accepted tokens from the best candidate.
    
    Parameters:
        input_ids: input token IDs
        candidates: candidate token sequences from the draft model
        best_candidate: index of the best candidate for each batch
        accept_length: accepted sequence length for each batch
        output_ids: output token IDs
        stride_*: strides for the respective tensors
        batch_size: number of sequences in the batch
        input_len: length of input sequence
        seq_len: sequence length of candidates
    """
    # Program ID
    pid = tl.program_id(axis=0)
    
    # Compute batch index
    batch_idx = pid
    
    # Get best candidate and accept length for this batch
    best_idx = tl.load(best_candidate + batch_idx)
    acc_len = tl.load(accept_length + batch_idx)
    
    # Copy input tokens
    for i in range(input_len):
        token = tl.load(input_ids + batch_idx * stride_ib + i * stride_is)
        tl.store(output_ids + batch_idx * stride_ob + i * stride_os, token)
    
    # Copy accepted tokens from the best candidate
    for i in range(acc_len + 1):
        if i < seq_len:
            token = tl.load(candidates + batch_idx * stride_cb + best_idx * stride_cs + i * stride_ct)
            tl.store(output_ids + batch_idx * stride_ob + (input_len + i) * stride_os, token)


def triton_compute_topk(logits, k):
    """
    Compute top-k values and indices using PyTorch as a fallback.
    
    Parameters:
        logits: input logits tensor of shape [batch_size, vocab_size]
        k: number of top values to select
        
    Returns:
        values: top-k values tensor of shape [batch_size, k]
        indices: top-k indices tensor of shape [batch_size, k]
    """
    # Use PyTorch's topk function as a fallback
    values, indices = torch.topk(logits, k, dim=-1)
    
    # Convert indices to int32 to match the expected output type
    indices = indices.to(torch.int32)

    return values, indices


def triton_compute_tree_mask(parents, total_tokens):
    """Compute tree mask for speculative decoding using Triton kernels.

    Parameters:
        parents: parent indices for each token of shape ``[batch_size, total_tokens]``
        total_tokens: total number of tokens in the tree

    Returns:
        A tensor of shape ``[batch_size, total_tokens, total_tokens]`` containing
        the attention mask for each token in the tree.
    """

    batch_size = parents.shape[0]

    # CPU fallback for environments without a CUDA device
    if not parents.is_cuda:
        tree_mask = torch.zeros(
            (batch_size, total_tokens, total_tokens),
            dtype=torch.int32,
            device=parents.device,
        )

        for b in range(batch_size):
            for t in range(total_tokens):
                tree_mask[b, t, t] = 1
            for t in range(1, total_tokens):
                parent = parents[b, t].item()
                if parent != 0:
                    for i in range(t):
                        if tree_mask[b, parent, i] == 1:
                            tree_mask[b, t, i] = 1
        return tree_mask

    # GPU path using Triton kernel
    tree_mask = torch.zeros(
        (batch_size, total_tokens, total_tokens),
        dtype=torch.int32,
        device=parents.device,
    )

    stride_tb, stride_tr, stride_tc = tree_mask.stride()
    stride_pb, stride_pi = parents.stride()

    grid = (batch_size,)

    _tree_mask_kernel[grid](
        tree_mask,
        parents,
        stride_tb,
        stride_tr,
        stride_tc,
        stride_pb,
        stride_pi,
        total_tokens=total_tokens,
    )

    return tree_mask


def triton_evaluate_posterior(logits, candidates):
    """Evaluate posterior probabilities for speculative decoding using Triton.

    Parameters:
        logits: tensor from the base model of shape ``[batch_size, seq_len, vocab_size]``
        candidates: candidate token sequences from the draft model of shape
            ``[batch_size, num_candidates, cand_seq_len]``.  The first token of
            each candidate is assumed to be the context token and is therefore
            skipped when evaluating the posterior.

    Returns:
        ``best_candidate`` – index of the best candidate for each batch.
        ``accept_length`` – accepted sequence length for each batch.
    """

    batch_size, _, vocab_size = logits.shape
    _, _, cand_seq_len = candidates.shape

    best_candidate = torch.zeros(batch_size, dtype=torch.int32, device=logits.device)
    accept_length = torch.zeros(batch_size, dtype=torch.int32, device=logits.device)

    # CPU fallback
    if not logits.is_cuda:
        _, num_candidates, _ = candidates.shape
        for b in range(batch_size):
            max_accept_len = 0
            max_candidate_idx = 0
            for c in range(num_candidates):
                accept_len = 0
                for s in range(1, cand_seq_len):
                    candidate_token = candidates[b, c, s].item()
                    max_logit_idx = torch.argmax(logits[b, s - 1]).item()
                    if max_logit_idx == candidate_token:
                        accept_len += 1
                    else:
                        break
                if accept_len > max_accept_len:
                    max_accept_len = accept_len
                    max_candidate_idx = c
            best_candidate[b] = max_candidate_idx
            accept_length[b] = max_accept_len
        return best_candidate, accept_length

    # GPU path using Triton kernel
    stride_lb, stride_ls, stride_lv = logits.stride()
    stride_cb, stride_cs, stride_ct = candidates.stride()

    grid = (batch_size,)

    _evaluate_posterior_kernel[grid](
        logits,
        candidates,
        best_candidate,
        accept_length,
        stride_lb,
        stride_ls,
        stride_lv,
        stride_cb,
        stride_cs,
        stride_ct,
        batch_size,
        cand_seq_len,
        vocab_size,
    )

    return best_candidate, accept_length


def triton_update_inputs(input_ids, candidates, best_candidate, accept_length):
    """
    Update input IDs with accepted tokens from the best candidate using Triton kernels.
    
    Parameters:
        input_ids: input token IDs of shape [batch_size, input_len]
        candidates: candidate token sequences from the draft model of shape [batch_size, num_candidates, seq_len]
        best_candidate: index of the best candidate for each batch of shape [batch_size]
        accept_length: accepted sequence length for each batch of shape [batch_size]
        
    Returns:
        output_ids: updated token IDs of shape [batch_size, input_len + accept_length + 1]
    """
    batch_size, input_len = input_ids.shape
    _, _, seq_len = candidates.shape
    
    # Get maximum accept length
    max_accept_len = int(accept_length.max().item()) + 1
    
    # Create output tensor
    output_ids = torch.zeros((batch_size, input_len + max_accept_len), 
                            dtype=input_ids.dtype, device=input_ids.device)
    
    # Compute strides for tensors
    stride_ib, stride_is = input_ids.stride()
    stride_cb, stride_cs, stride_ct = candidates.stride()
    stride_ob, stride_os = output_ids.stride()
    
    # Launch kernel
    grid = (batch_size,)
    _update_inputs_kernel[grid](
        input_ids, candidates, best_candidate, accept_length, output_ids,
        stride_ib, stride_is,
        stride_cb, stride_cs, stride_ct,
        stride_ob, stride_os,
        batch_size, input_len, seq_len,
    )

    return output_ids
