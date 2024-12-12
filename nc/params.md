# CORA + RANDOM

## Rwl-GCN



### ptb = 0.0



### ptb = 0.2

### ptb = 0.4

### ptb = 0.6

### ptb = 0.8

### ptb = 1.0


## Rwl-Hyla-SGC

python HyLa.py \
       -manifold poincare \
       -dataset cora \
       -attack random \
       -model hyla \
       -gnn_model SGC \
       -defence Rwl-GNN \
       -ptb_lvl 0.0 \
       -epochs 100 \
       -he_dim 16 \
       -lr_e 0.1 \
       -lr_c 0.01 \
       -lr_a 0.001 \
       -alpha 0.1 \
       -gamma 1.0 \
       -beta 0.1 \
       -tuned \
       -use_feats \

### ptb = 0.0



### ptb = 0.2

### ptb = 0.4

### ptb = 0.6

### ptb = 0.8

### ptb = 1.0