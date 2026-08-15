import os
import torch
from tqdm.auto import tqdm
from nltk.translate.bleu_score import corpus_bleu

def run_lstm_attention_pipeline(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    ru_vocab,
    device,
    num_epochs=5,
    save_dir='checkpoints/lstm_attention'
):
    os.makedirs(save_dir, exist_ok=True)
    scaler = torch.amp.GradScaler('cuda')
    current_step = 0

    for epoch in range(num_epochs):
        # --- TRAIN LOOP ---
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"LSTM Attention Epoch {epoch + 1}/{num_epochs}")

        for batch in pbar:
            src = batch[0].to(device)
            trg = batch[1].to(device)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                output = model(src, trg)
                output_flatten = output[1:].view(-1, output.shape[-1])
                trg_flatten = trg[1:].view(-1)
                loss = criterion(output_flatten, trg_flatten)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            current_step += 1
            current_loss = loss.item()
            epoch_loss += current_loss
            pbar.set_postfix({'batch_loss': f'{current_loss:.4f}'})

            # Checkpoint every 1000 steps
            if current_step % 1000 == 0:
                checkpoint = {
                    'epoch': epoch,
                    'step': current_step,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scaler_state': scaler.state_dict(),
                    'loss': current_loss
                }
                torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_step_{current_step}.pth'))

        avg_train_loss = epoch_loss / len(train_loader)

        # --- VALIDATION & BLEU LOOP ---
        model.eval()
        val_loss = 0
        references = []
        hypotheses = []

        PAD_IDX = ru_vocab.word2index['PAD']
        EOS_IDX = ru_vocab.word2index['EOS']

        with torch.no_grad():
            for batch in val_loader:
                src = batch[0].to(device)
                trg = batch[1].to(device)

                output = model(src, trg, teacher_forcing_ratio=0.0)
                output_flatten = output[1:].view(-1, output.shape[-1])
                trg_flatten = trg[1:].view(-1)
                val_loss += criterion(output_flatten, trg_flatten).item()

                top_predictions = output.argmax(dim=2).permute(1, 0)
                trg_permuted = trg.permute(1, 0)

                for i in range(trg_permuted.shape[0]):
                    ref_tokens = []
                    hyp_tokens = []

                    # Ground Truth
                    for token in trg_permuted[i][1:]:
                        token_idx = token.item()
                        if token_idx == EOS_IDX:
                            break
                        if token_idx != PAD_IDX:
                            ref_tokens.append(ru_vocab.index2word.get(token_idx, 'UNK'))

                    # Model Prediction
                    for token in top_predictions[i][1:]:
                        token_idx = token.item()
                        if token_idx == EOS_IDX:
                            break
                        if token_idx != PAD_IDX:
                            hyp_tokens.append(ru_vocab.index2word.get(token_idx, 'UNK'))

                    references.append([ref_tokens])
                    hypotheses.append(hyp_tokens)

        avg_val_loss = val_loss / len(val_loader)
        bleu_score = corpus_bleu(references, hypotheses) * 100

        print(f"Epoch {epoch + 1:02d} Completed | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | BLEU: {bleu_score:.2f}\n")

        # Save epoch checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'val_loss': avg_val_loss,
            'bleu': bleu_score
        }, os.path.join(save_dir, f'epoch_{epoch+1}.pth'))