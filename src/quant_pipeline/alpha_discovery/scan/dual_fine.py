from .dual_coarse import DualTileScanner


def fine_scanner(**kwargs) -> DualTileScanner:
    return DualTileScanner(bins=5, **kwargs)
