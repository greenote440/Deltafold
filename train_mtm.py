"""
Masked Topological Modeling (MTM) Training Loop
"""
import os
import glob
import re
import random
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from topotein import Topotein
from asymmetric_topotein import AsymmetricTopoNet
from train import PCCDataset, custom_collate, to_device, get_cluster_aware_split, DEVICE, PROC_DIR, CHECKPOINT_DIR, CLUSTER_TSV

# DEVICE is imported from train (honours --deltafold / DELTAFOLD_DEVICE); don't
# recompute it here or the forced-CUDA pin would be lost.
PROC_DIR = './data/hoan_processed'
CHECKPOINT_DIR = './checkpoints'
CLUSTER_TSV = './data/cluster.tsv'

class MTMReconstructionHead(nn.Module):
    """
    Projects the high-dimensional latent space representations back to 
    local structural alphabet probabilities.
    """
    def __init__(self, scalar_dim=128, num_classes=21):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(scalar_dim, scalar_dim),
            nn.LayerNorm(scalar_dim),
            nn.SiLU(),
            nn.Linear(scalar_dim, num_classes)
        )
        
    def forward(self, h0):
        return self.net(h0)

def worker_init_fn(worker_id):
    import torch
    torch.set_num_threads(1)

def train_mtm(model_type='topotein', epochs=30, batch_size=16, mask_ratio=0.25, lr=1e-4, accum_steps=1, use_reg=False, dataset_size=None, profile_train=False):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    loss_log_path = os.path.join(CHECKPOINT_DIR, 'mtm_training_losses.csv')
    
    train_files, val_files = get_cluster_aware_split(PROC_DIR, CLUSTER_TSV, split_ratio=0.8, seed=42)

    if dataset_size is not None:
        print(f"Limiting dataset to {dataset_size} training samples.")
        train_files = train_files[:dataset_size]
        val_size = int(dataset_size * (1.0 - 0.8) / 0.8) # Keep split ratio
        val_files = val_files[:val_size]

    if not train_files:
        print(f"No .pt files found in {PROC_DIR}. Run topotein_lifter.py first.")
        return

    # Optimized DataLoader: use persistent workers and conditional pin_memory (CUDA only)
    num_workers = max(1, (os.cpu_count() or 4) // 2)
    train_loader = DataLoader(PCCDataset(train_files), batch_size=batch_size, shuffle=True, collate_fn=custom_collate, num_workers=num_workers, prefetch_factor=2, persistent_workers=False, pin_memory=DEVICE.type == 'cuda', worker_init_fn=worker_init_fn)
    val_loader = DataLoader(PCCDataset(val_files), batch_size=batch_size, shuffle=False, collate_fn=custom_collate, num_workers=0, pin_memory=DEVICE.type == 'cuda')
    
    if model_type == 'asymmetric':
        model = AsymmetricTopoNet(scalar_dim=128).to(DEVICE)
    else:
        model = Topotein(scalar_dim=128).to(DEVICE)
        
    head = MTMReconstructionHead(scalar_dim=128, num_classes=21).to(DEVICE)
    
    trainable_params = list(model.parameters()) + list(head.parameters())
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss()
    
    start_epoch = 0
    best_loss = float('inf')
    last_ckpt_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_mtm_{model_type}_last.pth')

    if os.path.exists(last_ckpt_path):
        print(f"Loading checkpoint {last_ckpt_path}...")
        checkpoint = torch.load(last_ckpt_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        head.load_state_dict(checkpoint['head_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint.get('best_loss', float('inf'))
        print(f"Resuming from epoch {start_epoch + 1}")
    else:
        checkpoint_files = glob.glob(os.path.join(CHECKPOINT_DIR, 'checkpoint_ep*.pth'))
        if checkpoint_files:
            latest_ckpt = max(checkpoint_files, key=lambda f: int(re.search(r'_ep(\d+)\.pth', f).group(1)) if re.search(r'_ep(\d+)\.pth', f) else -1)
            print(f"Loading legacy checkpoint {latest_ckpt}...")
            checkpoint = torch.load(latest_ckpt, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            head.load_state_dict(checkpoint['head_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint.get('loss', float('inf'))
            print(f"Resuming from epoch {start_epoch + 1}")

    if start_epoch == 0:
        with open(loss_log_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'step', 'mtm_loss', 'val_loss'])
    
    print(f"Starting MTM Training ({model_type}): {len(train_files)} train, {len(val_files)} val samples.")
    
    for epoch in range(start_epoch, epochs):
        model.train()
        head.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for step, batch in enumerate(pbar):
            if batch is None:
                continue
            features = to_device(batch[0], DEVICE)
            
            r0 = features['rank0']
            n_res = r0['aa'].shape[0]
            
            if n_res == 0:
                continue
            
            num_mask = max(1, int(n_res * mask_ratio))
            mask_indices = random.sample(range(n_res), num_mask)
            
            orig_3di = r0['3di'][mask_indices].clone()
            target_classes = torch.argmax(orig_3di, dim=1)
            
            r0['3di'][mask_indices] = 0.0
            r0['aa'][mask_indices] = 0.0
            
            global_emb, h0 = model(features, return_nodes=True)
            masked_h0 = h0[mask_indices]
            logits = head(masked_h0)
            
            loss = criterion(logits, target_classes)
            scaled_loss = loss / accum_steps
            scaled_loss.backward()
            
            epoch_loss += loss.item()
            
            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                if DEVICE.type == 'mps':
                    # torch.linalg.vector_norm in clip_grad_norm_ causes massive CPU sync bottlenecks on MPS.
                    # Using clip_grad_value_ entirely bypasses the global reduction syncs.
                    torch.nn.utils.clip_grad_value_(trainable_params, clip_value=1.0)
                else:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)

                optimizer.step()
                optimizer.zero_grad()
            
            with open(loss_log_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, step + 1, f"{loss.item():.6f}", ""])
                
            mem_gb = _report_memory()
            pbar.set_postfix({'mtm': f"{loss.item():.4f}", 'ram': f"{mem_gb:.1f}GB"})

            if DEVICE.type == 'mps':
                del features, r0, loss, scaled_loss, global_emb, h0, masked_h0, logits
                gc.collect()
                torch.mps.empty_cache()
            
        # Validation Pass
        model.eval()
        head.eval()
        val_loss_epoch = 0.0
        with torch.no_grad():
            for v_batch in tqdm(val_loader, desc="Validation", leave=False):
                if v_batch is None: continue
                v_feat = to_device(v_batch[0], DEVICE)
                v_r0 = v_feat['rank0']
                v_n = v_r0['aa'].shape[0]
                if v_n == 0: continue
                v_num_mask = max(1, int(v_n * mask_ratio))
                v_mask_idx = random.sample(range(v_n), v_num_mask)
                v_targets = torch.argmax(v_r0['3di'][v_mask_idx], dim=1)
                v_r0['3di'][v_mask_idx] = 0.0
                v_r0['aa'][v_mask_idx] = 0.0
                _, v_h0 = model(v_feat, return_nodes=True)
                v_logits = head(v_h0[v_mask_idx])
                val_loss_epoch += criterion(v_logits, v_targets).item()
                del v_feat, v_r0, v_targets, v_h0, v_logits

        if DEVICE.type == 'mps':
            gc.collect()
            torch.mps.empty_cache()
        
        avg_train_loss = epoch_loss / len(train_loader)
        avg_val_loss = val_loss_epoch / len(val_loader) if len(val_loader) > 0 else 0.0
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1} Complete. Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        is_best = avg_val_loss < best_loss
        if is_best:
            best_loss = avg_val_loss

        best_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_mtm_{model_type}_best.pth')
        checkpoint_data = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'head_state_dict': head.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_val_loss,
            'best_loss': best_loss,
        }
        
        last_path = os.path.join(CHECKPOINT_DIR, 'checkpoint_mtm_last.pth')
        torch.save(checkpoint_data, last_path)
        
        if is_best:
            torch.save(checkpoint_data, best_path)
            print(f"New best checkpoint saved (Loss: {best_loss:.6f})")
            
        # Explicitly clear checkpoint objects and sync Metal cache
        del checkpoint_data
        if DEVICE.type == 'mps':
            gc.collect()
            torch.mps.empty_cache()

    return best_loss

    print("Training finished.")