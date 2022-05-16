for coef in 0.5 1.0
do
    python -m torch.distributed.launch --nproc_per_node 1 main_simmim.py --cfg configs/custom_contrastive/simmim_cont_pretrain__swin64__img96_window6__100ep.yaml --batch-size 128 --output experiment_results/ --amp-opt-level O0 --lambda_ $coef
done