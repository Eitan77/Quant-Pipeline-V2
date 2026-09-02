from .dual_coarse import DualTileScanner


def exact_scanner(**kwargs) -> DualTileScanner:
    return DualTileScanner(bins=10, **kwargs)
