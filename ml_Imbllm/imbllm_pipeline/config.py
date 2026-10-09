from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    """Central configuration; debug mode keeps the first run computationally small."""

    data_path: Path = Path("creditcard.csv") / "creditcard.csv"
    results_dir: Path = Path("results")
    target: str = "Class"
    q: float = 0.2
    test_size: float = 0.2
    seed: int = 42
    model_name: str = "distilgpt2"
    temperature: float = 0.7
    max_length: int = 128
    debug_mode: bool = True
    debug_generation_count: int = 100
    full_epochs: int = 5

    @property
    def epochs(self) -> int:
        return 1 if self.debug_mode else self.full_epochs

    def as_dict(self) -> dict:
        values = asdict(self)
        values.update({"data_path": str(self.data_path), "results_dir": str(self.results_dir)})
        values["epochs"] = self.epochs
        return values
