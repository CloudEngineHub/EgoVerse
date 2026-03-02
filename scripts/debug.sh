source /coc/flash7/bli678/Projects/EgoVerse/emimic/bin/activate

python egomimic/trainHydra.py --config-name=train_wm.yaml data=debug model=cosmos_policy trainer=debug logger=debug name=debug description=debug
python egomimic/trainHydra.py --config-name=train.yaml data=debug_hpt model=hpt_bc_flow_eva trainer=debug logger=debug name=debug description=debug
python egomimic/trainHydra.py --config-name=train_wm.yaml data=debug model=cosmos_policy trainer=ddp_cosmos_policy logger=debug name=debug description=debug