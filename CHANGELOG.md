## May 15 | 9:59

- Add some configurations facilitating training
- Modify some settings
- Remove some redundancies

## May 11 | 11:40

- Add 2 other metrics to Contrastive Learning: L1 and L2
- Add learning scheduler into training IGMC, change lr
- Replace some configs in training both Contrastive and IGMC

## May 10 | 10:33

- Fix error in creating data for training Regularization trick
- Add some configs to plug the trained MLP in Regularization trick to main training flow
- Some changes in IGMC model

## May 8 | 18:24

- Finish implementing **Regularization trick**

## May 6 | 20:23

- Add Regularization trick's implementation (ongoing)
- Replace GNN-LSPE by original GNN (ongoing)

## Apr 27 | 10:20

- Replace **TransformerEncoder** by **Hyper Mixer**
- Add **Distance Encoding** implementation but currently not used
- Fix error not calculating ARR loss with scenarios using **R-GCN-LSPE**

## Apr 4 | 10:15

- Add scenarios using _GatedGCN-LSPE_

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
