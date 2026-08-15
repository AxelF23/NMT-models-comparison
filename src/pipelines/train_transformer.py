import os
import torch
from tqdm.auto import tqdm

def run_transformer_pipeline(
    model,
    train_loader,
    optimizer,
    scheduler,
    criterion,
    device,
    num_epochs=5,
    save_dir='checkpoints/transformer'
):
    os.makedirs(save_dir, exist_ok=True)
    scaler = torch.amp.GradScaler('cuda')
    best_loss = float('inf')

    for epoch in range(num_epochs):
        # --- TRAIN LOOP ---
        model.train()
        epoch_loss = 0.0
        loop = tqdm(train_loader, desc=f"Transformer Epoch [{epoch+1}/{num_epochs}]")

        for src, trg in loop:
            src = src.to(device)
            trg = trg.to(device)
            trg_input = trg[:, :-1]
            trg_expected = trg[:, 1:]

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                output = model(src, trg_input)
                output = output.reshape(-1, output.shape[-1])
                trg_expected = trg_expected.reshape(-1)
                loss = criterion(output, trg_expected)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            if scheduler is not None:
                scheduler.step()

            current_loss = loss.item()
            epoch_loss += current_loss

            lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
            loop.set_postfix(loss=f"{current_loss:.4f}", lr=f"{lr:.6f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"--> Epoch {epoch+1} Completed. Average Train Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(save_dir, 'best_transformer.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, save_path)
            print(f"    [+] Model saved to {save_path} (Loss: {best_loss:.4f})")