python HylaMain.py \
       -gpu -1 \
       -manifold poincare \
       -dataset cora \
       -attack random \
       -model hyla \
       -gnn_model GNNGuard \
       -ptb_lvl 0.0 \
       -epochs 100 \
       -he_dim 16 \
       -lr_e 0.5 \
       -lr_c 0.01 \
       -lr_a 0.001 \
       -alpha 1.0 \
       -gamma 1.0 \
       -beta 0.1 \
       -tuned \
       -use_feats \
       -original \
&& echo -e '\a' # for beep sound on completion

# remove -use_feats option for airport, add -inductive option for inductive training on reddit