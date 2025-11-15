from pathlib import Path
from types import SimpleNamespace
from egomimic.scripts.eva_process.eva_to_lerobot import main as eva_main

def lerobot_job(
    *,
    raw_path: str | Path,
    output_dir: str | Path,
    dataset_name: str,
    arm: str,
    description: str = "",
    extrinsics: dict | list | None = None,
) -> None:
    raw_path = Path(raw_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()

    args = SimpleNamespace(
        name=dataset_name,
        raw_path=raw_path,
        dataset_repo_id=f"rpuns/{dataset_name}",
        fps=50,
        arm=arm,
        description=description,
        image_compressed=False,
        video_encoding=False,
        prestack=True,
        output_dir=output_dir,
        push=False,
        private=False,
        license="apache-2.0",
        nproc=12,
        nthreads=2,
        debug=False,
        extrinsics_key=extrinsics,
    )

    eva_main(args)
