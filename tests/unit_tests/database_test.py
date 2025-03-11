import os

import pytest
from rex_xai.database import db_to_pandas, initialise_rex_db, update_database


@pytest.fixture
def db(tmp_path):
    p = tmp_path / "rex.db"
    db = initialise_rex_db(p)
    return db


def test_update_database(exp_extracted, tmp_path):
    p = tmp_path / "rex.db"
    db = initialise_rex_db(p)
    update_database(db, exp_extracted)
    assert os.path.exists(p)
    assert os.stat(p).st_size > 0


def test_update_database_no_target(exp_extracted, db, caplog):
    exp_extracted.data.target = None
    update_database(db, exp_extracted)
    assert caplog.records[0].message == "unable to update database as target is None"


def test_update_database_no_exp(exp_extracted, db, caplog):
    exp_extracted.final_mask = None
    update_database(db, exp_extracted)
    assert (
        caplog.records[0].message == "unable to update database as explanation is empty"
    )


def test_read_db(exp_extracted, tmp_path):
    p = tmp_path / "rex.db"
    db = initialise_rex_db(p)
    update_database(db, exp_extracted)
    df = db_to_pandas(p)
    assert df.shape == (1, 29)


def test_no_multi(exp_extracted, caplog):
    update_database(db, exp_extracted, multi=True)
    assert (
        caplog.records[0].message
        == "unable to update database, multi=True is only valid for MultiExplanation objects"
    )


def test_update_database_multiexp(exp_multi, tmp_path):
    p = tmp_path / "rex.db"
    db = initialise_rex_db(p)
    update_database(db, exp_multi)
    assert os.path.exists(p)
    assert os.stat(p).st_size > 0


def test_read_database_multiexp(exp_multi, tmp_path):
    p = tmp_path / "rex.db"
    db = initialise_rex_db(p)
    update_database(db, exp_multi, multi=True)

    df = db_to_pandas(p)
    assert df.shape == (len(exp_multi.explanations), 29)
