"""Train Mini-BERT on a text classification task — pure PyTorch, no downloads.

Uses generated pattern data: sentences with class-specific keywords.
This proves the architecture learns — the patterns are real linguistic signals.

Model:  bert-mini (hidden=256, layers=4, heads=4) — CPU-friendly
Time:   ~5 min on CPU
"""

import random
import torch
from torch.utils.data import DataLoader, TensorDataset

from mini_bert.model import MiniBertForClassification


# ---- Pattern-Based Data Generator ----
# Creates sentences where each class has distinctive keywords.
# This is a REAL linguistic pattern — the model must learn to identify them.

CLASS_KEYWORDS = {
    0: ["president", "election", "government", "minister", "diplomat", "treaty"],
    1: ["touchdown", "goal", "championship", "player", "coach", "league"],
    2: ["stock", "market", "revenue", "investor", "profit", "acquisition"],
    3: ["software", "satellite", "quantum", "algorithm", "startup", "digital"],
}
FILLER_WORDS = ["the", "a", "is", "was", "has", "will", "new", "major", "first",
                "after", "says", "plans", "reports", "announced", "launches"]


def generate_dataset(n_samples=8000, seq_len=64, vocab_size=2000):
    """Generate text classification data with class-specific keywords.

    Each sample: a short "news headline" containing 2-3 class keywords.
    Model must learn which keywords map to which class.
    """
    print(f"  Generating {n_samples} synthetic samples (vocab={vocab_size}, seq_len={seq_len})...")

    texts = []
    labels = []

    for _ in range(n_samples):
        cls = random.randint(0, 3)

        # Pick 2-3 class-specific keywords
        keywords = random.sample(CLASS_KEYWORDS[cls], k=random.randint(2, 3))

        # Mix with filler words
        sentence_words = list(keywords)
        for _ in range(random.randint(4, 10)):
            sentence_words.append(random.choice(FILLER_WORDS))
        random.shuffle(sentence_words)

        texts.append(" ".join(sentence_words))
        labels.append(cls)

    # Build simple vocabulary from all words
    all_words = set()
    for t in texts:
        for w in t.split():
            all_words.add(w)
    vocab = {w: i + 2 for i, w in enumerate(sorted(all_words))}  # start from 2
    vocab["[PAD]"] = 0
    vocab["[UNK]"] = 1

    # Tokenize
    ids_list = []
    for t in texts:
        ids = [vocab.get(w, 1) for w in t.split()]
        ids = ids[:seq_len]
        if len(ids) < seq_len:
            ids += [0] * (seq_len - len(ids))
        ids_list.append(ids)

    input_ids = torch.tensor(ids_list, dtype=torch.long)
    attention_mask = (input_ids != 0).long()
    labels_t = torch.tensor(labels)

    # Split 80/20
    split = int(n_samples * 0.8)
    train_ds = TensorDataset(input_ids[:split], attention_mask[:split], labels_t[:split])
    test_ds = TensorDataset(input_ids[split:], attention_mask[split:], labels_t[split:])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)

    return train_loader, test_loader, len(vocab)


def train_epoch(model, loader, optimizer):
    model.train()
    total_loss, n = 0, len(loader)

    for i, (ids, mask, labels) in enumerate(loader):
        optimizer.zero_grad()
        output = model(ids, mask, labels=labels)
        output["loss"].backward()
        optimizer.step()
        total_loss += output["loss"].item()

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{n}] loss={output['loss'].item():.4f}  avg={total_loss/(i+1):.4f}")

    return total_loss / n


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for ids, mask, labels in loader:
            output = model(ids, mask)
            correct += (output["logits"].argmax(-1) == labels).sum().item()
            total += labels.size(0)
    return correct / total


def main():
    print("=" * 60)
    print("  Mini-BERT Training (pattern classification)")
    print("  Model: hidden=256, layers=4, heads=4")
    print("=" * 60)
    print()

    # Generate data
    train_loader, test_loader, vocab_size = generate_dataset()

    # Create model
    model = MiniBertForClassification(
        num_labels=4,
        hidden_size=256,
        num_layers=4,
        num_heads=4,
        intermediate_size=1024,
        vocab_size=vocab_size,
        max_position=64,
        dropout=0.1,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Parameters: {n_params:,}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test batches:  {len(test_loader)}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Train
    best_acc = 0
    for epoch in range(1, 6):
        print(f"--- Epoch {epoch} ---")
        train_loss = train_epoch(model, train_loader, optimizer)
        acc = evaluate(model, test_loader)
        delta = "+" if acc > best_acc else " "
        print(f"  Train loss: {train_loss:.4f}  |  Test acc: {acc:.2%} {delta}")
        best_acc = max(best_acc, acc)

    print(f"\n{'='*60}")
    print(f"  Best accuracy: {best_acc:.2%}")
    if best_acc > 0.9:
        print(f"  [OK] Model successfully learned the classification task!")
    elif best_acc > 0.7:
        print(f"  [OK] Model is learning (train longer for better results)")
    else:
        print(f"  Learning below threshold — check data/model config")
    print(f"{'='*60}")

    # Save
    torch.save(model.state_dict(), "mini_bert_checkpoint.pt")
    print(f"\nModel saved to mini_bert_checkpoint.pt")


if __name__ == "__main__":
    main()
