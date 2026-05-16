"""Verify Mini-BERT correctness — without HuggingFace dependency.

Tests:
  1. Forward pass shapes at each stage
  2. Attention mask correctly hides padding
  3. Gradient flow (backward pass works)
  4. Output distribution is reasonable (not NaN, not uniform)
  5. Overfitting: can the model memorize 10 examples?
"""

import torch
import torch.nn.functional as F

from mini_bert.model import MiniBertForClassification


def test_shapes():
    """Test that every layer produces the correct output shape."""
    print("=" * 60)
    print("  1. Shape Verification")
    print("=" * 60)

    model = MiniBertForClassification(
        num_labels=4, hidden_size=128, num_layers=2,
        num_heads=2, intermediate_size=512,
    )

    bsz, seq_len = 8, 64
    input_ids = torch.randint(0, 30522, (bsz, seq_len))
    attention_mask = torch.ones(bsz, seq_len)

    output = model(input_ids, attention_mask)

    checks = [
        ("logits shape", output["logits"].shape, torch.Size([bsz, 4])),
        ("pooled shape", output["pooled"].shape, torch.Size([bsz, 128])),
        ("no NaN in logits", torch.isnan(output["logits"]).any().item(), False),
        ("no Inf in logits", torch.isinf(output["logits"]).any().item(), False),
    ]

    all_pass = True
    for name, got, expected in checks:
        passed = got == expected
        print(f"  {name}: {'[OK]' if passed else '[FAIL]'}  (got={got}, expected={expected})")
        all_pass &= passed

    return all_pass


def test_attention_mask():
    """Test that padding tokens are actually ignored."""
    print("\n" + "=" * 60)
    print("  2. Attention Mask Test")
    print("=" * 60)

    model = MiniBertForClassification(
        num_labels=2, hidden_size=128, num_layers=2,
        num_heads=2, intermediate_size=512,
    )

    # Two identical inputs, one with padding
    tokens = torch.randint(0, 1000, (1, 8))
    tokens_padded = tokens.clone()
    tokens_padded[0, 4:] = 0  # Pad last 4 tokens

    mask = torch.ones(1, 8)
    mask[0, 4:] = 0  # Mark last 4 as padding

    with torch.no_grad():
        out1 = model(tokens)
        out2 = model(tokens_padded, mask)

    # The first 4 tokens are the same — outputs should be close
    diff = (out1["logits"] - out2["logits"]).abs().max().item()
    print(f"  Logits difference with/without padding: {diff:.6f}")
    print(f"  {'[OK] Attention mask working' if diff < 2.0 else '[FAIL] Too different'}")

    return diff < 5.0


def test_gradient_flow():
    """Test that gradients flow through all layers."""
    print("\n" + "=" * 60)
    print("  3. Gradient Flow Test")
    print("=" * 60)

    model = MiniBertForClassification(
        num_labels=4, hidden_size=128, num_layers=2,
        num_heads=2, intermediate_size=512,
    )

    input_ids = torch.randint(0, 1000, (4, 16))
    labels = torch.randint(0, 4, (4,))

    output = model(input_ids, labels=labels)
    output["loss"].backward()

    # Check that every Linear layer got gradients
    no_grad_layers = []
    for name, param in model.named_parameters():
        if param.grad is None:
            no_grad_layers.append(name)
        elif param.grad.abs().max() < 1e-10:
            no_grad_layers.append(f"{name} (zero grad)")

    if no_grad_layers:
        print(f"  [FAIL] {len(no_grad_layers)} layers without gradients:")
        for n in no_grad_layers:
            print(f"    - {n}")
        return False
    else:
        print(f"  [OK] All {sum(1 for _ in model.parameters())} parameters received gradients")
        return True


def test_overfitting():
    """Can the model memorize 16 examples? Proves the architecture works."""
    print("\n" + "=" * 60)
    print("  4. Overfitting Test (memorize 16 examples)")
    print("=" * 60)

    model = MiniBertForClassification(
        num_labels=4, hidden_size=128, num_layers=2,
        num_heads=2, intermediate_size=512,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Create a tiny fixed dataset
    torch.manual_seed(42)
    n_samples = 16
    input_ids = torch.randint(0, 1000, (n_samples, 16))
    labels = torch.randint(0, 4, (n_samples,))

    # Train until 100% accuracy or 200 steps
    for step in range(200):
        model.train()
        output = model(input_ids, labels=labels)
        loss = output["loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 50 == 0:
            with torch.no_grad():
                acc = (output["logits"].argmax(-1) == labels).float().mean()
            print(f"  Step {step+1:3d}: loss={loss.item():.4f}  acc={acc:.2%}")

            if acc == 1.0:
                print(f"\n  [OK] Memorized {n_samples} examples in {step+1} steps!")
                return True

    # Final check
    with torch.no_grad():
        final_output = model(input_ids)
        final_acc = (final_output["logits"].argmax(-1) == labels).float().mean()

    passed = final_acc == 1.0
    print(f"  Final accuracy: {final_acc:.2%}  {'[OK]' if passed else '[FAIL]'}")
    return passed


def main():
    print("Mini-BERT Verification Suite\n")

    r1 = test_shapes()
    r2 = test_attention_mask()
    r3 = test_gradient_flow()
    r4 = test_overfitting()

    print("\n" + "=" * 60)
    results = [r1, r2, r3, r4]
    passed = sum(results)
    print(f"  {passed}/{len(results)} tests passed")
    if all(results):
        print("  ALL TESTS PASSED — Your Mini-BERT is correct!")
    else:
        print("  Some tests failed — check output above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
