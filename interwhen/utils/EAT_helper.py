from vllm import SamplingParams
import torch
import numpy as np  

def compute_diversity(responses, device='cuda:2'):
    # Assert responses is a list of string
    if not isinstance(responses, list) or not all(isinstance(r, str) for r in responses):
        raise ValueError("Responses must be a list of strings.")
    from sentence_transformers import SentenceTransformer
    embd_model = SentenceTransformer("all-mpnet-base-v2")
    embd_model.to(device)
    embeddings = embd_model.encode(responses)
    similarities = embd_model.similarity(embeddings, embeddings).cpu().numpy()
    diagonal_mask = np.eye(embeddings.shape[0], dtype=bool)
    off_diagonal_elements = np.where(diagonal_mask, 0, similarities)
    sum_off_diagonal = np.mean(off_diagonal_elements)
    return 1 - np.mean(sum_off_diagonal)


def numpy_softmax(x, axis=-1):
    """Compute softmax using numpy with double precision for better numerical stability"""
    # Convert to double precision
    x_double = x.astype(np.float64)
    # Subtract max for numerical stability
    x_shifted = x_double - np.max(x_double, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True) 

def get_token_rank(logits, token_id):
    """Get the rank of a specific token in the logit space (1 = highest logit)"""
    # Count how many tokens have higher logits
    higher_count = np.sum(logits > logits[token_id])
    # Rank is higher_count + 1 (1-indexed)
    return int(higher_count) + 1

def entropy(probabilities, base=2):
    """
    Computes the entropy of a categorical distribution using NumPy.

    Parameters:
    probabilities (list or np.ndarray): A list or array of probabilities for each category.
    base (int): The logarithmic base to use, default is 2.

    Returns:
    float: The entropy of the distribution.
    """
    probabilities = np.array(probabilities)
    
    # if not np.isclose(probabilities.sum(), 1.0):
    #     raise ValueError("Probabilities must sum to 1.")
    
    # Filter out zero probabilities to avoid log(0)
    non_zero_probs = probabilities[probabilities > 0]
    entropy = -np.sum(non_zero_probs * np.log(non_zero_probs) / np.log(base))
    
    return entropy


def calculate_conditional_perplexity(model, tokenizer, prompt, responses):
    # Check model is a huggingface model
    if not hasattr(model, 'generate'):
        raise ValueError("Model must be a Hugging Face model with a generate method.")
    pps = []
    for response in responses:
        full_text = prompt + response
        
        # Tokenize both parts
        prompt_tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
        full_tokens = tokenizer(full_text, return_tensors="pt")["input_ids"].to(model.device)
        
        # Get the length of prompt in tokens
        prompt_length = prompt_tokens.size(1)
        
        with torch.no_grad():
            # Get model predictions for the full sequence
            outputs = model(full_tokens)
            logits = outputs.logits
            
            # Calculate loss only for the response part (after prompt)
            # Shift logits and labels for next token prediction
            shift_logits = logits[..., prompt_length:-1, :].contiguous()
            shift_labels = full_tokens[..., prompt_length + 1:].contiguous()
            # print(shift_logits.shape)
            # Calculate cross-entropy loss
            # print(shift_logits)
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                        shift_labels.view(-1))
            
            # Calculate perplexity
            perplexity = torch.exp(loss)
        pps.append(perplexity.item())
        
    return pps


def calculate_conditional_perplexity_fast(model, tokenizer, prompt, responses):
    if not hasattr(model, 'generate'):
        raise ValueError("Model must be a Hugging Face model with a generate method.")
    
    pps = []
    
    # Tokenize prompt once
    prompt_tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
    prompt_length = prompt_tokens.size(1)
    
    # Get KV cache for prompt
    with torch.no_grad():
        prompt_outputs = model(prompt_tokens, use_cache=True)
        past_key_values = prompt_outputs.past_key_values
    
    for response in responses:
        past_key_values.crop(prompt_length)  # Crop past key values to the length of the prompt
        # Tokenize response
        response_tokens = tokenizer(response, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
        response_length = response_tokens.size(1)
        
        with torch.no_grad():
            # Forward pass for response using cached prompt KV
            outputs = model(
                input_ids=response_tokens,
                past_key_values=past_key_values,
                use_cache=False
            )
            
            # Calculate loss for response tokens
            logits = outputs.logits
            
            # Shift for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = response_tokens[..., 1:].contiguous()
            # print(shift_logits)
            # Calculate cross-entropy loss
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                          shift_labels.view(-1))
            
            # Calculate perplexity
            perplexity = torch.exp(loss)
        
        pps.append(perplexity.item())
    
    return pps


def compute_entropy(model, tokenizer, context, kv_cache=None, max_context_length=8192):
    # Compute the next token entropy, and the entropy when appended stop thinking token
    with torch.no_grad():
        inputs_with_EoT_token = tokenizer(context, return_tensors="pt").to(model.device)['input_ids']
        
        # Truncate to max_context_length if needed (keep the last max_context_length tokens)
        # This is important for long prompts that exceed the model's context window
        if inputs_with_EoT_token.shape[1] > max_context_length:
            inputs_with_EoT_token = inputs_with_EoT_token[:, -max_context_length:]
        
        # print(kv_cache.get_seq_length())
        # print(inputs_with_EoT_token.shape)
        outputs = model(
            inputs_with_EoT_token,
            output_hidden_states=False,
            return_dict=True,
            use_cache=False,
            past_key_values=None
        )
        next_token_score = outputs.logits[0].float().detach().cpu().numpy()
        next_token_score = numpy_softmax(next_token_score[-1])
        result = entropy(next_token_score)
        
        # Explicit memory cleanup to prevent OOM
        del inputs_with_EoT_token
        del outputs
        del next_token_score
        torch.cuda.empty_cache()
        
        return result
    

def compute_entropy_ngram(model, tokenizer, context, N=10, K=20, kv_cache=None):
    with torch.no_grad():
        inputs_with_EoT_token = tokenizer(context, return_tensors="pt").to(model.device)['input_ids']
        scores = []
        input_shape = kv_cache.get_seq_length()
        for _ in range(N):
            rollouts = model.generate(
                inputs_with_EoT_token,
                max_new_tokens=K,
                do_sample=True,
                temperature=0.6,
                use_cache=True,
                past_key_values=kv_cache,
                output_scores=True,
                return_dict_in_generate=True,
                top_k=None,
                top_p=None,
            )
            O = torch.stack(list(rollouts.scores)).squeeze().detach().cpu().numpy()
            if K != 1:
                if O.shape[0] != K:
                    continue
            scores.append(
                torch.stack(list(rollouts.scores)).squeeze().detach().cpu().numpy()
            )
            # print(torch.stack(list(rollouts.scores)).squeeze().detach().cpu().numpy().shape)
            kv_cache.crop(input_shape)
        scores = np.array(scores)
        next_token_score = numpy_softmax(scores, -1)
        E = np.apply_along_axis(entropy, -1, next_token_score)
        return np.cumsum(E, -1).mean(0)


def compute_bigram_entropy_vectorized(model, tokenizer, prefix, num_mc_sample=10, batch_size=1, kv_cache=None):
    # assert batch_size == 1
    with torch.no_grad():
        inputs = tokenizer(prefix, return_tensors="pt").to(model.device)['input_ids']
        if kv_cache is not None:
            inputs = inputs[:, kv_cache.get_seq_length():]
        outputs = model(
            inputs,
            output_hidden_states=False,
            return_dict=True,
            use_cache=True,
            past_key_values=kv_cache
        )
        if kv_cache is not None:
            past_key_values = kv_cache
        else:
            past_key_values = outputs.past_key_values
        original_length = past_key_values.get_seq_length()
        past_key_values.batch_repeat_interleave(batch_size)
        
        next_token_logits = outputs.logits[:, -1, :]
        next_token_probs = torch.nn.functional.softmax(next_token_logits, dim=-1).cpu().numpy().squeeze()
        entropy_first_token = entropy(next_token_probs)
        # attention_mask = inputs.attention_mask.expand(batch_size, -1)
        # attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1), dtype=torch.long, device=model.device)], dim=1)
        entropy_second_token = []
        for _ in range(num_mc_sample // batch_size):
            # past_key_values.crop(inputs['input_ids'].shape[1])
            past_key_values.crop(original_length)
            sampled_tokens = np.random.choice(
                len(next_token_probs),
                size=batch_size,
                p=next_token_probs,
                replace=True
            )
            sampled_tokens_tensor = torch.tensor(sampled_tokens, device=model.device).unsqueeze(1)

            outputs_second_token = model(
                input_ids=sampled_tokens_tensor,
                # attention_mask=attention_mask,
                past_key_values=past_key_values,
                output_hidden_states=False,
                return_dict=True
            )
            
            second_token_logits = outputs_second_token.logits.squeeze(1)
            second_token_probs = torch.nn.functional.softmax(second_token_logits, dim=-1).cpu().numpy()
            # Compute entropy for each sample's distribution in a vectorized way
            entropy_second_token.append(np.apply_along_axis(entropy, 1, second_token_probs))

        entropy_second_token = np.concatenate(entropy_second_token)
        bigram_entropy = entropy_first_token + np.mean(entropy_second_token)
        
    return entropy_first_token, bigram_entropy


def get_multi_completion_hf(
        model,
        context,
        num_generations=1,
        temperature=0.6,
        top_p=0.95,
        top_k=50,
        max_tokens=2000
    ):
    raise NotImplementedError(
        "This function is not implemented yet. Please use the vLLM version instead."
    )


def get_multi_completion_vllm(
        model,
        context,
        num_generations=1,
        temperature=0.6,
        top_p=0.95,
        top_k=-1,
        max_tokens=2000,
        seed=1,
        sample_params = None,
    ):
    if not sample_params:
        sample_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            n=num_generations,
            seed=seed
        )
    
    contiunation = model.generate(
        context, sample_params,
        use_tqdm=False
    )
    return [
        o.text for o in contiunation[0].outputs
    ]


def exponential_moving_average(data, timescale):
    ema = np.zeros_like(data)
    ema[0] = data[0]
    
    for i in range(1, len(data)):
        ema[i] = timescale * data[i] + (1 - timescale) * ema[i - 1]
    
    return ema


def exponential_moving_variance(data, timescale, ema_0=0.0, normalize=False):
    ema_mean = exponential_moving_average(data, timescale)
    ema_variance = np.zeros_like(data)
    ema_variance[0] = ema_0
    
    for i in range(1, len(data)):
        deviation = data[i] - ema_mean[i]
        ema_variance[i] = timescale * (deviation ** 2) + (1 - timescale) * ema_variance[i - 1]

    return ema_variance