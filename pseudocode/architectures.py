"""
architectures.py  —  Conceptual pseudocode (abstraction level 4–5/10)
Neural Machine Translation: Token by Token

Covers three progressively more powerful architectures:
  1. Seq2Seq Encoder  — shared by both RNN and LSTM variants
  2. Seq2Seq Decoder  — with optional cosine-similarity attention
  3. Seq2Seq          — the full encoder-decoder wrapper
  4. TransformerTranslator — parallel self-attention architecture

This file is NOT executable. It captures design intent and data-flow
at a conceptual level. Exact tensor dimensions, initialisation
strategies, and framework-specific calls are intentionally omitted.
"""


# ─────────────────────────────────────────────────────────────────────────────
#  1.  RECURRENT ENCODER
# ─────────────────────────────────────────────────────────────────────────────

class RecurrentEncoder:
    """
    Maps a source token sequence to a fixed-size context representation.

    Supports two recurrent cell types (plain RNN or LSTM).  The final
    hidden state is projected through a small feed-forward bottleneck so
    its dimensionality matches whatever the decoder expects.

    Design note:
        For LSTM, both the hidden state AND the cell state are returned —
        the cell state bypasses the projection and is passed raw to the
        decoder to preserve internal memory continuity.

    Layers (in order):
        Embedding      — integer token IDs → dense vectors
        Dropout        — regularises the embedding before the recurrent pass
        Recurrent      — RNN or LSTM, processes the full source sequence
        Projection     — Linear → ReLU → Linear → Tanh
                         maps encoder hidden dim to decoder hidden dim
    """

    def __init__(self, vocab_size, recurrent_type):
        self.embedding  = token_embedding_table(vocab_size, ...)
        self.recurrent  = recurrent_layer(recurrent_type, ...)   # "RNN" or "LSTM"
        self.projection = sequential(linear(...), relu(), linear(...), tanh())
        self.dropout    = dropout_layer(...)

    def forward(self, source_tokens):
        """
        1. Embed source tokens; apply dropout.
        2. Feed the full embedded sequence into the recurrent layer,
           collecting all intermediate hidden states (encoder_outputs)
           and the final hidden state.
        3. Project the final hidden state through the bottleneck.
        4. Return:
             encoder_outputs — all T hidden states; used by attention
             projected_hidden — decoder initialisation state
             (cell_state)    — raw cell state, LSTM only, not projected
        """
        embedded        = self.dropout(self.embedding(source_tokens))
        encoder_outputs, final_state = self.recurrent(embedded)
        projected_hidden = self.projection(extract_last_hidden(final_state))
        return encoder_outputs, projected_hidden


# ─────────────────────────────────────────────────────────────────────────────
#  2.  RECURRENT DECODER  (with optional cosine attention)
# ─────────────────────────────────────────────────────────────────────────────

class RecurrentDecoder:
    """
    Generates target tokens one at a time, conditioned on the encoder.

    Without attention: the decoder relies solely on the encoder's final
    hidden state (bottleneck).  Each step receives only the previously
    predicted token embedding.

    With attention: at every decode step the decoder queries all encoder
    hidden states using cosine similarity, computes a weighted sum
    (context vector), concatenates it with the token embedding, and
    projects the fusion down before feeding it into the recurrent cell.
    This breaks the fixed-size bottleneck — the decoder sees a fresh
    source summary at every step.

    Layers (in order):
        Embedding           — target token IDs → dense vectors
        Dropout             — regularises the target embedding
        Recurrent           — RNN or LSTM cell (same type as encoder)
        OutputProjection    — hidden dim → vocab size, log-softmax
        AttentionFusion     — (context ‖ embedding) → embedding dim   [attention only]
    """

    def __init__(self, target_vocab_size, recurrent_type, use_attention):
        self.embedding         = token_embedding_table(target_vocab_size, ...)
        self.recurrent         = recurrent_layer(recurrent_type, ...)
        self.output_projection = sequential(linear(...), log_softmax(dim=vocab_axis))
        self.dropout           = dropout_layer(...)
        if use_attention:
            self.context_fusion = linear(encoder_hidden_dim + embedding_dim, embedding_dim)

    def compute_attention(self, decoder_hidden, encoder_outputs):
        """
        Cosine-similarity attention over all encoder positions.

        Intuition:
            Instead of dot-product attention (which is magnitude-sensitive),
            cosine similarity measures the DIRECTION of agreement between
            the decoder query and each encoder key — more stable early in
            training when magnitudes are arbitrary.

        Steps:
            1. Reshape decoder hidden state to align with encoder output dims.
            2. Compute cosine similarity between the decoder state and each
               of the T encoder hidden states — produces a score per position.
            3. Normalise scores to a probability distribution (softmax).
            4. Return attention weights of shape (batch, 1, T) for use
               in a batched weighted sum.
        """
        query            = reshape(decoder_hidden, ...)   # align dims for broadcast
        scores           = cosine_similarity(query, encoder_outputs, along_hidden_axis)
        attention_weights = softmax(scores, along_sequence_axis)
        return expand_dims(attention_weights, ...)        # shape ready for bmm

    def step(self, input_token, hidden_state, encoder_outputs):
        """
        Single decode step.

        1. Embed input token; apply dropout.
        2. If attention:
               weights        = compute_attention(hidden_state, encoder_outputs)
               context_vector = batched_dot(weights, encoder_outputs)
               fused_input    = context_fusion(concat(context_vector, embedding))
           Else:
               fused_input    = embedding
        3. Feed fused_input and hidden_state into the recurrent cell.
        4. Project recurrent output to vocab logits via output_projection.
        5. Return log-probability distribution over target vocab, new hidden state.
        """
        embedded   = self.dropout(self.embedding(input_token))

        if self.use_attention:
            weights        = self.compute_attention(extract_hidden(hidden_state), encoder_outputs)
            context_vector = batched_weighted_sum(weights, encoder_outputs)
            fused_input    = self.context_fusion(concat(context_vector, embedded))
        else:
            fused_input    = embedded

        recurrent_out, new_hidden = self.recurrent(fused_input, hidden_state)
        log_probs = self.output_projection(recurrent_out)
        return log_probs, new_hidden


# ─────────────────────────────────────────────────────────────────────────────
#  3.  SEQ2SEQ WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class Seq2SeqModel:
    """
    Combines RecurrentEncoder and RecurrentDecoder into a full
    sequence-to-sequence translation pipeline.

    Training uses greedy decoding (argmax at each step feeds next step);
    teacher forcing is not shown here for brevity.
    """

    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source_tokens):
        """
        1. Run encoder over the full source sequence.
           Receive: encoder_outputs (all T states) + initial decoder hidden state.

        2. Initialise decoder input with the <SOS> token (first token of source).

        3. Loop for each target position:
               log_probs, hidden = decoder.step(current_input, hidden, encoder_outputs)
               Store log_probs.
               Next input = argmax(log_probs)  [greedy]

        4. Stack all per-step log-probability tensors → output shape (batch, T, vocab).
        """
        encoder_outputs, decoder_hidden = self.encoder(source_tokens)

        current_input = start_of_sequence_token(source_tokens)
        all_outputs   = []

        for _ in range(target_sequence_length):
            log_probs, decoder_hidden = self.decoder.step(
                current_input, decoder_hidden, encoder_outputs
            )
            all_outputs.append(log_probs)
            current_input = argmax(log_probs)

        return stack(all_outputs, along_time_axis)


# ─────────────────────────────────────────────────────────────────────────────
#  4.  TRANSFORMER TRANSLATOR
# ─────────────────────────────────────────────────────────────────────────────

class TransformerTranslator:
    """
    Encoder-only Transformer for sequence-to-sequence translation.

    Architecture overview:
        Input tokens → Embedding (word + positional) →
        Multi-Head Self-Attention (with Add & LayerNorm) →
        Position-wise Feedforward (with Add & LayerNorm) →
        Linear projection → token logits

    Key departure from Seq2Seq:
        No recurrence.  The entire source sequence is processed in ONE
        parallel pass.  Every position attends to every other position
        simultaneously — there is no fixed-size bottleneck.

    Layers:
        word_embedding      — maps token IDs to dense vectors
        position_embedding  — learned per-position offsets (same dim)
        attention_heads     — parallel projection + scaled dot-product attention,
                              one set of Q/K/V projections per head
        attention_norm      — layer normalisation after the attention sublayer
        feedforward_expand  — large linear expansion (up-projection)
        feedforward_contract — linear contraction back to model dim
        feedforward_norm    — layer normalisation after the FFN sublayer
        output_projection   — maps model dim to target vocabulary size
    """

    def __init__(self, source_vocab_size, target_vocab_size, num_heads):
        self.word_embedding       = token_embedding_table(source_vocab_size, ...)
        self.position_embedding   = positional_embedding_table(max_sequence_length, ...)

        # One Q, K, V projection matrix per head
        self.q_projections = [linear(...) for _ in range(num_heads)]
        self.k_projections = [linear(...) for _ in range(num_heads)]
        self.v_projections = [linear(...) for _ in range(num_heads)]
        self.output_merge  = linear(num_heads * head_dim, model_dim)

        self.attention_norm    = layer_norm(model_dim)
        self.feedforward_expand  = linear(model_dim, expanded_dim)
        self.feedforward_contract = linear(expanded_dim, model_dim)
        self.feedforward_norm    = layer_norm(model_dim)
        self.output_projection   = linear(model_dim, target_vocab_size)

    def embed(self, token_ids):
        """
        Fuse semantic and positional information.

        Word embedding captures meaning; positional embedding tells the
        model WHERE in the sequence each token sits (necessary because
        self-attention is inherently permutation-equivariant — without
        position info, the model cannot distinguish word order).

        Steps:
            1. Look up word embedding for each token ID.
            2. Look up learned positional embedding for each position index.
            3. Add the two — element-wise sum, same shape as each embedding.
        Returns:
            Fused embedding tensor of shape (batch, sequence_length, model_dim).
        """
        token_repr    = self.word_embedding(token_ids)
        position_repr = self.position_embedding(position_indices_for(token_ids))
        return token_repr + position_repr

    def multi_head_attention(self, x):
        """
        Parallel scaled dot-product attention across multiple subspaces.

        Intuition:
            Different heads can specialise in different relationship types
            (syntactic, semantic, positional).  Each head projects the
            same input into a lower-dimensional Q/K/V subspace, runs
            scaled dot-product attention independently, then all head
            outputs are concatenated and linearly merged.

        Per head:
            Q = x @ W_q,  K = x @ W_k,  V = x @ W_v
            head_output = softmax(Q @ K.T / sqrt(head_key_dim)) @ V

        After all heads:
            concat(head_outputs) → output_merge linear → residual add → LayerNorm

        Steps:
            1. For each head: compute Q, K, V projections from x.
            2. Compute scaled dot-product attention scores.
            3. Softmax-normalise scores; apply to V.
            4. Concatenate all head outputs.
            5. Project merged heads back to model_dim.
            6. Add residual (x) and apply LayerNorm.

        Returns:
            Attended representation, same shape as input x.
        """
        head_outputs = []
        for head_idx in range(self.num_heads):
            Q = x @ self.q_projections[head_idx]
            K = x @ self.k_projections[head_idx]
            V = x @ self.v_projections[head_idx]
            scores      = Q @ transpose(K) / sqrt(head_key_dim)
            attention   = softmax(scores, along_sequence_axis)
            head_outputs.append(attention @ V)

        merged    = concat(head_outputs, along_feature_axis)
        projected = self.output_merge(merged)
        return self.attention_norm(x + projected)    # residual + LayerNorm

    def feedforward_layer(self, x):
        """
        Position-wise two-layer feedforward network.

        Applied independently and identically at every sequence position.
        The large intermediate expansion provides nonlinear capacity that
        linear self-attention cannot express on its own.

        Steps:
            1. Expand: linear → ReLU activation.
            2. Contract: linear back to model_dim.
            3. Add residual (x) and apply LayerNorm.

        Returns:
            Transformed representation, same shape as input x.
        """
        expanded  = relu(self.feedforward_expand(x))
        contracted = self.feedforward_contract(expanded)
        return self.feedforward_norm(x + contracted)  # residual + LayerNorm

    def final_layer(self, x):
        """
        Map the model's internal representation to token logits.

        Projects each position's hidden vector to a score over the target
        vocabulary.  No activation — raw logits returned; cross-entropy
        loss applies its own log-softmax internally during training.

        Returns:
            Logit tensor of shape (batch, sequence_length, target_vocab_size).
        """
        return self.output_projection(x)

    def forward(self, source_tokens):
        """
        Full forward pass (encoder-only Transformer).

        1. embed(source_tokens)           — fuse word + positional signals
        2. multi_head_attention(embedded) — attend across all positions
        3. feedforward_layer(attended)    — position-wise nonlinear transform
        4. final_layer(transformed)       — project to vocab logits

        Returns:
            Logit tensor; argmax over vocab axis gives predicted tokens.
        """
        embedded     = self.embed(source_tokens)
        attended     = self.multi_head_attention(embedded)
        transformed  = self.feedforward_layer(attended)
        return self.final_layer(transformed)
