import os

def test_surface(exp_extracted, tmp_path):
    p = tmp_path / "surface.png"
    exp_extracted.surface_plot(path=p)

    assert os.path.exists(p)
    assert os.stat(p).st_size > 0


def test_heatmap(exp_extracted, tmp_path):
    p = tmp_path / "heatmap.png"
    exp_extracted.heatmap_plot(path=p)

    assert os.path.exists(p)
    assert os.stat(p).st_size > 0


def test_save_exp(exp_extracted, tmp_path):
    p = tmp_path / "exp.png"
    exp_extracted.save(path=p)

    assert os.path.exists(p)
    assert os.stat(p).st_size > 0
