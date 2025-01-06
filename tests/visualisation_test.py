from syrupy.extensions.image import PNGImageSnapshotExtension


def test_surface(exp_extracted, tmp_path, snapshot):
    p = tmp_path / "surface.png"
    exp_extracted.surface_plot(path=p)

    with open(p, "rb") as ifile:
        assert snapshot(extension_class=PNGImageSnapshotExtension) == ifile.read()


def test_heatmap(exp_extracted, tmp_path, snapshot):
    p = tmp_path / "heatmap.png"
    exp_extracted.heatmap_plot(path=p)

    with open(p, "rb") as ifile:
        assert snapshot(extension_class=PNGImageSnapshotExtension) == ifile.read()


def test_save_exp(exp_extracted, tmp_path, snapshot):
    p = tmp_path / "exp.png"
    exp_extracted.save(path=p)

    with open(p, "rb") as ifile:
        assert snapshot(extension_class=PNGImageSnapshotExtension) == ifile.read()
