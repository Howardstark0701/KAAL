"""Adversarial text attacks for HuggingFace sequence classification models.

kaal/attacks/text_attack.py

Supports any model loadable via:
    AutoTokenizer.from_pretrained(...)
    AutoModelForSequenceClassification.from_pretrained(...)

Examples: bert-base-uncased, distilbert-base-uncased, roberta-base,
          any fine-tuned sentiment / classification checkpoint.

Two attack methods
------------------
1. token_substitution_attack()
   Uses the model's attention weights to identify the most attended
   tokens in each input, then replaces them with synonyms from a
   hardcoded dictionary. No WordNet, no internet lookup.

2. embedding_perturbation_attack()
   Adds L-inf bounded noise directly to the embedding layer output,
   then passes the perturbed embeddings through the rest of the model.
   This is the text analogue of FGSM in pixel space.

All tensor operations run on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Synonym dictionary (~50 common words, no WordNet dependency)
# ---------------------------------------------------------------------------

_SYNONYMS: dict[str, list[str]] = {
    # sentiment / opinion words
    "good":       ["great", "fine", "solid", "decent"],
    "great":      ["good", "excellent", "superb", "fine"],
    "bad":        ["poor", "awful", "terrible", "weak"],
    "terrible":   ["awful", "horrible", "dreadful", "bad"],
    "awful":      ["terrible", "horrible", "dreadful", "bad"],
    "excellent":  ["outstanding", "superb", "great", "brilliant"],
    "amazing":    ["incredible", "fantastic", "extraordinary", "remarkable"],
    "wonderful":  ["fantastic", "marvelous", "splendid", "great"],
    "horrible":   ["dreadful", "awful", "terrible", "appalling"],
    "beautiful":  ["lovely", "stunning", "gorgeous", "attractive"],
    "ugly":       ["hideous", "unattractive", "unsightly", "plain"],
    "happy":      ["glad", "pleased", "joyful", "content"],
    "sad":        ["unhappy", "sorrowful", "miserable", "gloomy"],
    "angry":      ["furious", "irate", "annoyed", "enraged"],
    "funny":      ["humorous", "amusing", "comical", "witty"],
    "boring":     ["dull", "tedious", "monotonous", "uninteresting"],
    "interesting":["engaging", "fascinating", "captivating", "intriguing"],
    "important":  ["significant", "crucial", "vital", "essential"],
    "difficult":  ["hard", "challenging", "tough", "demanding"],
    "easy":       ["simple", "straightforward", "effortless", "uncomplicated"],
    "fast":       ["quick", "rapid", "swift", "speedy"],
    "slow":       ["sluggish", "gradual", "leisurely", "unhurried"],
    "big":        ["large", "huge", "enormous", "substantial"],
    "small":      ["tiny", "little", "minor", "compact"],
    "strong":     ["powerful", "robust", "sturdy", "forceful"],
    "weak":       ["feeble", "frail", "fragile", "powerless"],
    "smart":      ["intelligent", "clever", "bright", "sharp"],
    "stupid":     ["foolish", "dumb", "unintelligent", "ignorant"],
    "new":        ["fresh", "novel", "recent", "modern"],
    "old":        ["ancient", "aged", "antique", "dated"],
    # common verbs
    "said":       ["stated", "mentioned", "noted", "remarked"],
    "think":      ["believe", "consider", "suppose", "reckon"],
    "know":       ["understand", "realize", "recognize", "grasp"],
    "want":       ["desire", "wish", "need", "seek"],
    "like":       ["enjoy", "appreciate", "prefer", "favor"],
    "love":       ["adore", "cherish", "treasure", "admire"],
    "hate":       ["despise", "detest", "loathe", "dislike"],
    "get":        ["obtain", "acquire", "receive", "gain"],
    "show":       ["demonstrate", "reveal", "display", "exhibit"],
    "make":       ["create", "produce", "build", "form"],
    "give":       ["provide", "offer", "supply", "deliver"],
    "take":       ["grab", "obtain", "acquire", "seize"],
    "use":        ["employ", "utilize", "apply", "leverage"],
    "help":       ["assist", "support", "aid", "facilitate"],
    "try":        ["attempt", "endeavor", "strive", "seek"],
    # common nouns
    "problem":    ["issue", "challenge", "difficulty", "concern"],
    "result":     ["outcome", "finding", "conclusion", "effect"],
    "work":       ["effort", "task", "job", "labor"],
}


# ---------------------------------------------------------------------------
# TextAttackResult
# ---------------------------------------------------------------------------

@dataclass
class TextAttackResult:
    """Result of a text adversarial attack."""

    attack_type: str
    """'token_substitution' or 'embedding_perturbation'."""

    success_rate: float
    """Fraction of samples where the attack caused a class change (0–1)."""

    avg_confidence_on_target: float
    """Average model confidence on target_class after attack (0–1)."""

    n_samples: int
    """Number of text samples processed."""

    plain_english: str
    """One factual sentence describing the outcome. No drama."""


# ---------------------------------------------------------------------------
# TextAttacker
# ---------------------------------------------------------------------------

class TextAttacker:
    """Adversarial attacker for HuggingFace text classification models.

    Supports any AutoModelForSequenceClassification-compatible checkpoint:
    BERT, DistilBERT, RoBERTa, ALBERT, etc.

    All operations run on CPU.

    Usage:
        attacker = TextAttacker("distilbert-base-uncased-finetuned-sst-2-english")
        result = attacker.token_substitution_attack(
            texts=["This movie was great!", "Absolutely terrible film."],
            target_class=0,
            n_substitutions=3,
        )
        print(result.success_rate, result.plain_english)
    """

    def __init__(self, model_name_or_path: str) -> None:
        """Load tokenizer and model.

        Args:
            model_name_or_path: HuggingFace model identifier or local path.
                                 Examples: "bert-base-uncased",
                                 "distilbert-base-uncased-finetuned-sst-2-english"

        Raises:
            ImportError: transformers is not installed.
        """
        try:
            from transformers import (
                AutoTokenizer,
                AutoModelForSequenceClassification,
            )
        except ImportError:
            raise ImportError(
                "The 'transformers' library is required for TextAttacker.\n"
                "Install it with:\n"
                "    pip install transformers>=4.35.0\n"
                "You may also need:\n"
                "    pip install sentencepiece  # for some tokenizers"
            )

        self.model_name = model_name_or_path

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            output_attentions=True,   # needed for token_substitution_attack
        )
        self._model.eval()
        # Force CPU — no GPU assumptions
        self._model = self._model.cpu()

    # ------------------------------------------------------------------
    # Method 1: Token substitution via attention weights
    # ------------------------------------------------------------------

    def token_substitution_attack(
        self,
        texts: list[str],
        target_class: int,
        n_substitutions: int = 3,
    ) -> TextAttackResult:
        """Replace high-attention tokens with synonyms to fool the model.

        Algorithm:
            1. Tokenize input and run forward pass with output_attentions=True.
            2. Average attention weights across all heads and all layers to
               get a per-token importance score.
            3. Find the top-n_substitutions tokens that:
               (a) are in the synonym dictionary (lowercased), AND
               (b) have the highest mean attention weight.
            4. Replace each selected token with its first available synonym.
            5. Re-tokenize the modified text and run inference.
            6. Record whether the prediction changed.

        Args:
            texts:           List of input strings.
            target_class:    Class index to steer the model toward.
            n_substitutions: Max number of token replacements per text.

        Returns:
            TextAttackResult with success rate and avg target confidence.
        """
        successes     = 0
        total_conf    = 0.0
        n             = len(texts)

        for text in texts:
            # ── Original prediction ───────────────────────────────────────
            orig_pred, _ = self._predict(text)
            orig_class   = int(orig_pred.argmax().item())

            # ── Attention-weighted token importance ───────────────────────
            enc = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            with torch.no_grad():
                outputs = self._model(**enc)

            # attentions: tuple of (1, n_heads, seq_len, seq_len) per layer
            # Average over layers and heads → (seq_len,) importance scores
            attn_layers = outputs.attentions  # may be None for some models
            if attn_layers:
                # Stack: (n_layers, 1, n_heads, seq_len, seq_len)
                stacked = torch.stack(attn_layers, dim=0)
                # Mean over layers, batch, heads → (seq_len, seq_len)
                mean_attn = stacked.mean(dim=(0, 1, 2))
                # Column mean: how much each token is attended to
                token_importance = mean_attn.mean(dim=0)  # (seq_len,)
            else:
                # Fallback: uniform importance (no attention available)
                seq_len = enc["input_ids"].shape[1]
                token_importance = torch.ones(seq_len)

            # ── Map token ids to words ────────────────────────────────────
            input_ids  = enc["input_ids"][0].tolist()
            tokens     = self._tokenizer.convert_ids_to_tokens(input_ids)

            # Sort tokens by descending importance, skip special tokens
            special    = {self._tokenizer.cls_token, self._tokenizer.sep_token,
                          self._tokenizer.pad_token, None}
            candidates = [
                (float(token_importance[i]), i, tok)
                for i, tok in enumerate(tokens)
                if tok not in special and not tok.startswith("##")
            ]
            candidates.sort(reverse=True)  # highest attention first

            # ── Apply substitutions ───────────────────────────────────────
            modified_text = text
            substituted   = 0

            for _, tok_idx, token in candidates:
                if substituted >= n_substitutions:
                    break
                # Clean token (remove leading ## for subword pieces)
                word = token.lstrip("#").lower()
                if word in _SYNONYMS:
                    synonym = _SYNONYMS[word][0]
                    # Case-preserving replacement
                    original_word = token.lstrip("#")
                    replacement   = _preserve_case(original_word, synonym)
                    # Replace first occurrence in the text (word-boundary aware)
                    modified_text = _replace_word(modified_text, original_word, replacement)
                    substituted  += 1

            # ── Post-attack prediction ────────────────────────────────────
            adv_pred, adv_conf = self._predict(modified_text)
            adv_class          = int(adv_pred.argmax().item())
            conf_on_target     = float(adv_pred[target_class].item())

            if adv_class != orig_class:
                successes += 1
            total_conf += conf_on_target

        success_rate = successes / n if n > 0 else 0.0
        avg_conf     = total_conf / n if n > 0 else 0.0

        return TextAttackResult(
            attack_type="token_substitution",
            success_rate=round(success_rate, 4),
            avg_confidence_on_target=round(avg_conf, 4),
            n_samples=n,
            plain_english=_build_plain_english(
                "token substitution (attention-guided synonym replacement)",
                success_rate, avg_conf, target_class, n,
            ),
        )

    # ------------------------------------------------------------------
    # Method 2: Embedding perturbation (text-domain FGSM analogue)
    # ------------------------------------------------------------------

    def embedding_perturbation_attack(
        self,
        texts: list[str],
        target_class: int,
        epsilon: float = 0.1,
    ) -> TextAttackResult:
        """Add L-inf bounded noise to the embedding layer output.

        Algorithm:
            1. Tokenize input.
            2. Get the embedding layer output (word + position embeddings).
            3. Add uniform noise in [-epsilon, +epsilon] to each embedding
               vector (same shape, L-inf bounded).
            4. Pass the perturbed embeddings through the rest of the model
               by replacing the embedding layer call via a forward hook.
            5. Record prediction change.

        This is the continuous relaxation of a text attack — the perturbed
        embeddings do not map back to discrete tokens, so it measures the
        model's sensitivity in embedding space rather than producing a
        human-readable adversarial text.

        Args:
            texts:        List of input strings.
            target_class: Class index to steer the model toward.
            epsilon:      L-inf noise bound applied to embeddings. Default 0.1.

        Returns:
            TextAttackResult with success rate and avg target confidence.
        """
        successes  = 0
        total_conf = 0.0
        n          = len(texts)

        for text in texts:
            # ── Original prediction ───────────────────────────────────────
            orig_pred, _ = self._predict(text)
            orig_class   = int(orig_pred.argmax().item())

            # ── Tokenize ──────────────────────────────────────────────────
            enc = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )

            # ── Extract base embeddings ───────────────────────────────────
            emb_module = self._get_embedding_module()
            with torch.no_grad():
                base_embs = emb_module(enc["input_ids"])  # (1, seq_len, hidden)

            # ── Add L-inf bounded noise ───────────────────────────────────
            noise        = torch.empty_like(base_embs).uniform_(-epsilon, epsilon)
            perturbed    = base_embs + noise              # (1, seq_len, hidden)

            # ── Forward pass with perturbed embeddings ────────────────────
            # Use a hook to inject perturbed embeddings instead of running
            # the embedding layer again
            perturbed_out: list[torch.Tensor] = []

            def _emb_hook(module, inp, out):
                # Replace the embedding output with our perturbed version
                perturbed_out.append(perturbed)
                return perturbed

            hook = emb_module.register_forward_hook(_emb_hook)
            try:
                with torch.no_grad():
                    outputs = self._model(**enc)
            finally:
                hook.remove()

            logits         = outputs.logits[0]           # (num_labels,)
            probs          = F.softmax(logits, dim=0)
            adv_class      = int(probs.argmax().item())
            conf_on_target = float(probs[target_class].item())

            if adv_class != orig_class:
                successes += 1
            total_conf += conf_on_target

        success_rate = successes / n if n > 0 else 0.0
        avg_conf     = total_conf / n if n > 0 else 0.0

        return TextAttackResult(
            attack_type="embedding_perturbation",
            success_rate=round(success_rate, 4),
            avg_confidence_on_target=round(avg_conf, 4),
            n_samples=n,
            plain_english=_build_plain_english(
                f"embedding perturbation (L-inf noise eps={epsilon})",
                success_rate, avg_conf, target_class, n,
            ),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _predict(self, text: str) -> tuple[torch.Tensor, float]:
        """Run inference and return (softmax_probs, confidence_of_top_class).

        Returns:
            probs: (num_labels,) float tensor, sum = 1.0
            conf:  float, confidence of the top-1 predicted class.
        """
        enc = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            outputs = self._model(**enc)
        probs = F.softmax(outputs.logits[0], dim=0)
        conf  = float(probs.max().item())
        return probs, conf

    def _get_embedding_module(self) -> torch.nn.Module:
        """Return the word embedding module from the underlying model.

        Works for BERT, DistilBERT, RoBERTa, ALBERT, and most variants
        that expose a `.embeddings` sub-module with a `.word_embeddings`
        or `.embeddings` child.

        Falls back to the top-level `.embeddings` attribute if a more
        specific child is not found.
        """
        base = self._model.base_model   # strips the classification head

        # Try common embedding attribute names
        for attr in ("embeddings", "embedding_layer", "embed_tokens"):
            module = getattr(base, attr, None)
            if module is not None and isinstance(module, torch.nn.Module):
                return module

        # Last resort: walk and return the first Embedding layer found
        for name, module in base.named_modules():
            if isinstance(module, torch.nn.Embedding):
                return module

        raise RuntimeError(
            f"Could not locate an embedding module in '{self.model_name}'.\n"
            "→ This model architecture may not be supported by embedding_perturbation_attack."
        )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _preserve_case(original: str, replacement: str) -> str:
    """Apply the case style of `original` to `replacement`."""
    if original.isupper():
        return replacement.upper()
    if original[0].isupper():
        return replacement.capitalize()
    return replacement.lower()


def _replace_word(text: str, original: str, replacement: str) -> str:
    """Replace the first case-insensitive occurrence of `original` in text."""
    import re
    # Word-boundary aware, case-insensitive
    pattern = re.compile(r'\b' + re.escape(original) + r'\b', re.IGNORECASE)
    result, n_subs = pattern.subn(replacement, text, count=1)
    return result if n_subs > 0 else text


def _build_plain_english(
    method: str,
    success_rate: float,
    avg_conf: float,
    target_class: int,
    n: int,
) -> str:
    """One factual sentence. No drama, no exclamation marks."""
    return (
        f"{method.capitalize()} attack on {n} text sample"
        f"{'s' if n != 1 else ''} achieved a {success_rate:.0%} success rate "
        f"with average confidence {avg_conf:.2f} on target class {target_class}."
    )
