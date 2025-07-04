import pandas as pd

from egomimic.scripts.evaluation.eval import Eval

class BlindEval(Eval):
    def __init__(
        self,
        **kwargs
    ):
        self.models = kwargs.get("models", None)
        self.blind_eval_path = kwargs.get("blind_eval_path")
        if self.models is not None and self.blind_eval_path is not None:
            raise ValueError("Only models or blind_eval_path should be specified")
        if self.models is None and self.blind_eval_path is None:
            raise ValueError("One of models or blind eval path needs to be specified")
        
        if self.models:
            
            
        