python egomimic/trainHydra.py \
    --config-name=train.yaml \
    data=scene_diversity/scene_eval \
    logger.wandb.project=everse_scene_diversity_fold_clothes \
    name=eval-fold-clothes-scene-diversity \
    description=scenes-4-time-30 \
    train=false \
    validate=true \
    ckpt_path="/coc/cedarp-dxu345-0/bli678/EgoVerse/logs/fold_clothes/scene_diversity/scenes-4-time-30_2026-01-26_13-18-12/checkpoints/epoch_epoch\=1399.ckpt"