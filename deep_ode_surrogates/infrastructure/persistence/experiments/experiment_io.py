from pathlib import Path

from deep_ode_surrogates.application.config.experiment import ExperimentConfig


def save_experiment_config(config: ExperimentConfig, output_dir: Path) -> Path:
    """
    Sauvegarde une expérience dans un fichier JSON horodaté.

    Args:
        config: Configuration complète de l'expérience.
        output_dir: Dossier de destination.

    Returns:
        Path du fichier créé.
    """
    filepath = output_dir / "training_conf.json"
    filepath.write_text(config.model_dump_json(indent=2), encoding="utf-8")


def load_experiment(path: str | Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate_json(Path(path).read_text(encoding="utf-8"))
