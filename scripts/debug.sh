# python egomimic/trainHydra.py \
#     --config-name=train.yaml \
#     data=cotrain_s3 \
#     model=hpt_cotrain_flow_shared_head \
#     trainer=debug \
#     logger=debug \
#     name=debug \
#     description=robot_bc \


python egomimic/trainHydra.py --config-name=train_wm.yaml data=debug model=cosmos_policy trainer=debug logger=debug name=debug description=debug 