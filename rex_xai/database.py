#!/usr/bin/env python
from datetime import datetime
import sqlalchemy as sa
import zlib
from sqlalchemy import Boolean, Float, String, create_engine
from sqlalchemy import Column, Integer, Unicode
from ast import literal_eval

import pandas as pd
from sqlalchemy.orm import sessionmaker, DeclarativeBase
import numpy as np

from rex_xai.config import CausalArgs, Strategy
from rex_xai.prediction import Prediction
from rex_xai.extraction import Explanation
from rex_xai.multi_explanation import MultiExplanation


def _dataframe(db, table):
    return pd.read_sql_table(table, f"sqlite:///{db}")


def _to_numpy(buffer, shape, dtype):
    return np.frombuffer(zlib.decompress(buffer), dtype=dtype).reshape(shape)


def db_to_pandas(db, dtype=np.float32, table="rex"):
    """for interactive use"""
    df = _dataframe(db, table=table)

    df["responsibility"] = df.apply(
        lambda row: _to_numpy(
            row["responsibility"], literal_eval(row["responsibility_shape"]), dtype
        ),
        axis=1,
    )

    df["explanation"] = df.apply(
        lambda row: _to_numpy(
            row["explanation"], literal_eval(row["explanation_shape"]), np.bool_
        ),
        axis=1,
    )

    return df


def update_database(
    db,
    target: Prediction,
    explanation: Explanation,
    time_taken,
    total_passing,
    total_failing,
    max_depth_reached,
    avg_box_size,
    multi=False,
    multi_no=None,
):
    if isinstance(explanation, Explanation):
        add_to_database(
            db,
            explanation.args,
            target.classification,
            target.confidence,
            explanation.target_map,
            explanation.final_mask.detach().cpu().numpy(),
            time_taken,
            total_passing,
            total_failing,
            max_depth_reached,
            avg_box_size,
        )

    elif isinstance(explanation, MultiExplanation):
        add_to_database(
            db,
            explanation.args,
            target.classification,
            target.confidence,
            explanation.target_map,
            explanation.explanations[multi_no].detach().cpu().numpy(),  # type: ignore
            time_taken,
            total_passing,
            total_failing,
            max_depth_reached,
            avg_box_size,
            multi=multi,
            multi_no=multi_no,
        )


def add_to_database(
    db,
    args: CausalArgs,
    target,
    confidence,
    responsibility,
    explanation,
    time_taken,
    passing,
    failing,
    depth_reached,
    avg_box_size,
    multi=False,
    multi_no=None,
):
    if multi:
        id = hash(str(datetime.now().time()) + str(multi_no))
    else:
        id = hash(str(datetime.now().time()))

    responsibility_shape = responsibility.shape
    explanation_shape = explanation.shape
    print(responsibility_shape)
    print(explanation_shape)
    print(responsibility.dtype)
    print(explanation.dtype)

    object = DataBaseEntry(
        id,
        args.path,
        target,
        confidence,
        responsibility,
        responsibility_shape,
        explanation,
        explanation_shape,
        time_taken,
        depth_reached=depth_reached,
        avg_box_size=avg_box_size,
        tree_depth=args.tree_depth,
        search_limit=args.search_limit,
        iters=args.iters,
        min_size=args.min_box_size,
        distribution=str(args.distribution),
        distribution_args=str(args.distribution_args),
    )
    # if object is not None:
    object.multi = multi
    object.multi_no = multi_no
    object.passing = passing
    object.failing = failing
    object.total_work = passing + failing
    if args.strategy == Strategy.Spatial:
        object.spatial_radius = args.spatial_radius
        object.spatial_eta = args.spatial_eta
    if args.strategy == Strategy.MultiSpotlight:
        object.method = str(args.strategy)
        object.spotlights = args.spotlights
        object.spotlight_size = args.spotlight_size
        object.spotlight_eta = args.spotlight_eta
        object.obj_function = str(args.spotlight_objective_function)

    db.add(object)
    db.commit()


class Base(DeclarativeBase):
    pass


class RankingType(sa.types.TypeDecorator):
    impl = sa.types.Text

    def process_bind_param(self, value, _):
        return str(value)

    def process_result_value(self, value, _):
        return value


class NumpyType(sa.types.TypeDecorator):
    impl = sa.types.LargeBinary

    def process_bind_param(self, value, _):
        value = value.copy(order="C")
        return zlib.compress(value, 9)

    def process_result_value(self, value, _):
        if value is not None:
            # this still needs to be reshaped to recreate the original matrix
            return np.frombuffer(zlib.decompress(value), dtype=np.float32)


class DataBaseEntry(Base):
    __tablename__ = "rex"
    id = Column(Integer, primary_key=True)
    path = Column(Unicode(100))
    target = Column(Integer)
    confidence = Column(Float)
    time = Column(Float)
    responsibility = Column(NumpyType)
    responsibility_shape = Column(Unicode)
    total_work = Column(Integer)
    passing = Column(Integer)
    failing = Column(Integer)
    explanation = Column(NumpyType)
    explanation_shape = Column(Unicode)
    multi = Column(Boolean)
    multi_no = Column(Integer)

    # causal specific columns
    depth_reached = Column(Integer)
    avg_box_size = Column(Float)
    tree_depth = Column(Integer)
    search_limit_per_iter = Column(Integer)
    iters = Column(Integer)
    min_size = Column(Integer)
    distribution = Column(String)
    distribution_args = Column(String)

    # explanation specific columns
    spatial_radius = Column(Integer)
    spatial_eta = Column(Float)

    # spotlight columns
    method = Column(String)
    spotlights = Column(Integer)
    spotlight_size = Column(Integer)
    spotlight_eta = Column(Float)
    obj_function = Column(String)

    def __init__(
        self,
        id,
        path,
        target,
        confidence,
        responsibility,
        responsibility_shape,
        explanation,
        explanation_shape,
        time_taken,
        passing=None,
        failing=None,
        total_work=None,
        multi=False,
        multi_no=None,
        depth_reached=None,
        avg_box_size=None,
        tree_depth=None,
        search_limit=None,
        iters=None,
        min_size=None,
        distribution=None,
        distribution_args=None,
        initial_radius=None,
        radius_eta=None,
        method=None,
        spotlights=None,
        spotlight_size=None,
        spotlight_eta=None,
        obj_function=None,
    ):
        self.id = id
        self.path = path
        self.target = target
        self.confidence = confidence
        self.responsibility = responsibility
        self.responsibility_shape = str(responsibility_shape)
        self.explanation = explanation
        self.explanation_shape = str(explanation_shape)
        self.time = time_taken
        self.total_work = total_work
        self.passing = passing
        self.failing = failing
        # multi status
        self.multi = multi
        self.multi_no = multi_no
        # causal
        self.depth_reached = depth_reached
        self.avg_box_size = avg_box_size
        self.tree_depth = tree_depth
        self.search_limit = search_limit
        self.iters = iters
        self.min_size = min_size
        self.distribution = distribution
        self.distribution_args = distribution_args
        # spatial
        self.spatial_radius = initial_radius
        self.spatial_eta = radius_eta
        self.method = method
        # spotlights
        self.spotlights = spotlights
        self.spotlight_size = spotlight_size
        self.spotlight_eta = spotlight_eta
        self.obj_function = obj_function


def initialise_rex_db(name, echo=False):
    engine = create_engine(f"sqlite:///{name}", echo=echo)
    Base.metadata.create_all(engine, tables=[DataBaseEntry.__table__], checkfirst=True)  # type: ignore
    Session = sessionmaker(bind=engine)
    s = Session()
    return s
