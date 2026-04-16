from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC_ROOT = ROOT / "src"
WINCLIP_ROOT = ROOT / "baselines" / "WinCLIP"
MVTEC_ROOT = ROOT / "data" / "mvtec"

sys.path.insert(0, str(SRC_ROOT))

from utils.image_loader import MVTecStyleDataset
from master_benchmark import sample_seeded_support_samples

sys.path.insert(0, str(WINCLIP_ROOT))
from run_winclip_mvtec_cpu_benchmark import collect_train_good_paths, sample_support_items


def patchcore_selected_names(category: str, n_shot: int, seed: int):
    dataset = MVTecStyleDataset(
        root_dir=str(MVTEC_ROOT),
        category=category,
        is_train=True,
        img_size=256,
    )
    selected = sample_seeded_support_samples(dataset.samples, n_shot=n_shot, seed=seed)
    return [Path(path).name for path, _ in selected]


def winclip_selected_names(category: str, n_shot: int, seed: int):
    train_good_paths = collect_train_good_paths(str(MVTEC_ROOT), category)
    selected = sample_support_items(train_good_paths, n_shot=n_shot, seed=seed)
    return [Path(path).name for path, _, _, _ in selected]


def main():
    category = "bottle"
    n_shot = 5
    seed = 111

    patchcore_names = patchcore_selected_names(category, n_shot, seed)
    winclip_names = winclip_selected_names(category, n_shot, seed)

    print(f"Category={category}, N={n_shot}, Seed={seed}")
    print("PatchCore first 3:", patchcore_names[:3])
    print("WinCLIP  first 3:", winclip_names[:3])
    print("Exact match:", patchcore_names == winclip_names)


if __name__ == "__main__":
    main()
