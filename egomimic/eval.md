eval.yaml file
- (ckpt_path) the model checkpoint being loaded. This is instantiated inside the eval class
- (multirun_cfg) the multirun_cfg allows you load the model and also datasets of the multirun_cfg
- (eval_path) the dir of the evals
- (datasets) null (no dataset being created), multirun (multirun dataset), specify the dataset you want with the key
- (debug) here so that you can pass it into eval class using $ in hydra.

