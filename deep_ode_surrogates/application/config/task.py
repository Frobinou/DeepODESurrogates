from pydantic import BaseModel


class TaskConfig(BaseModel):
    x_names: list[str] = []
    y_names: list[str] = []

    @property
    def input_dim(self) -> int:
        return len(self.x_names)

    @property
    def output_dim(self) -> int:
        return len(self.y_names)
