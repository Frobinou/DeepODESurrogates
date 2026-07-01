from pydantic import BaseModel


class EvaluatorConfig(BaseModel):
    use_mse: bool = True
    use_ode: bool = False
    use_trajectory: bool = False


class EvaluationConfig(BaseModel):
    use_mse: bool = True
    use_trajectory: bool = True
