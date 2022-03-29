## Mar 29 | 14:35

- Add **TransformerEncoder** layer to before GNN
- Replace **Adam** by **AdamW**
- Set SEED and n\*epoch for **yahoo music** are 1 and 20
- Increase common k for RWPE from 20 to 40

## Mar 22 | 16:05

- Implement **Old-R-GCN-LSPE** that inherits **R-GCN** implementation from pyg 1.4.2
- Implement **EdgeAugment** (not used yet)

## Mar 20 | 21:09

- Discard **GatedGCN** and variants, back to **R-GCN**
- Implement **R-GCN-LSPE** and **Fast-R-GCN-LSPE**

## Mar 18 | 23:05

- Tailor different training scripts for each dataset
- Add implementation for **RGatedGCNLayer**, and **FastRGatedGCNLayer** (still error)

## Mar 13 | 12:04

- Add `GatedGCNLSPELayer` and code for scenario [5, 6, 7, 8]
- Back to pyg ver **1.4.2**
