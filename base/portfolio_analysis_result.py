import pandas as pd

from base.model import Model


class PortfolioAnalysisResult:

    def __init__(self, model: Model, evaluation: pd.Series):
        self.model = model
        self.evaluation = evaluation

    def get_score(self) -> float:
        return self.evaluation.sum()
