import subprocess
import sys

# Import scorpus in a fresh interpreter where torch cannot be found, as if it were not installed.
NO_TORCH = """
import importlib.abc, sys

class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ModuleNotFoundError(f"No module named {name!r}")

sys.meta_path.insert(0, NoTorch())
"""


def test_scorpus_imports_without_torch():
    code = NO_TORCH + (
        "import scorpus, scorpus.cli, scorpus.pp\n"
        "from scorpus.loaders import MetadataIndex, ExpressionReader, load_corpus, concat\n"
        "assert 'torch' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_torch_loaders_need_torch():
    code = NO_TORCH + (
        "try:\n"
        "    from scorpus.loaders import build_loader\n"
        "except ModuleNotFoundError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('build_loader imported without torch')\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
