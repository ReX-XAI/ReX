#!/usr/bin/env python
from datetime import datetime
import zlib
import torch as tt
import sqlalchemy as sa
from sqlalchemy import Boolean, Float, String, create_engine
from sqlalchemy import Column, Integer, Unicode
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from ast import literal_eval

import pandas as pd
import numpy as np

from rex_xai.logger import logger
from rex_xai.config import CausalArgs, Strategy
from rex_xai.extraction import Explanation
from rex_xai.multi_explanation import MultiExplanation


def _dataframe(db, table):
    return pd.read_sql_table(table, f"sqlite:///{db}")


def _to_numpy(buffer, shape, dtype):
    return np.frombuffer(zlib.decompress(buffer), dtype=dtype).reshape(shape)


def db_to_pandas(db, dtype=np.float32, table="rex", process=True):
    """for interactive use"""
    df = _dataframe(db, table=table)

    if process:
        df["responsibility"] = df.apply(
            lambda row: _to_numpy(
                row["responsibility"], literal_eval(row["responsibility_shape"]), dtype
            ),
            axis=1,
        )
        #
        df["explanation"] = df.apply(
            lambda row: _to_numpy(
                row["explanation"], literal_eval(row["explanation_shape"]), np.bool_
            ),
            axis=1,
        )

    return df


def update_database(
    db,
    explanation: Explanation | MultiExplanation,  # type: ignore
    time_taken=None,
    multi=False,
):
    target_map = explanation.target_map

    if isinstance(target_map, tt.Tensor):
        target_map = target_map.detach().cpu().numpy()

    target = explanation.data.target
    if target is None:
        logger.warning("unable to update database as target is None")
        return
    classification = int(target.classification)  # type: ignore

    if not multi:
        final_mask = explanation.final_mask
        if explanation.final_mask is None:
            logger.warning("unable to update database as explanation is empty")
            return
        if isinstance(final_mask, tt.Tensor):
            final_mask = final_mask.detach().cpu().numpy()

        explanation_confidence = explanation.explanation_confidence

        add_to_database(
            db,
            explanation.args,
            classification,
            target.confidence,
            target_map,
            final_mask,
            explanation_confidence,
            time_taken,
            explanation.run_stats["total_passing"],
            explanation.run_stats["total_failing"],
            explanation.run_stats["max_depth_reached"],
            explanation.run_stats["avg_box_size"],
        )

    else:
        if type(explanation) is not MultiExplanation:
            logger.warning(
                "unable to update database, multi=True is only valid for MultiExplanation objects"
            )
            return
        else:
            for c, final_mask in enumerate(explanation.explanations):
                if isinstance(final_mask, tt.Tensor):
                    final_mask = final_mask.detach().cpu().numpy()
                add_to_database(
                    db,
                    explanation.args,
                    classification,
                    target.confidence,
                    target_map,
                    final_mask,
                    explanation.explanation_confidences[c],
                    time_taken,
                    explanation.run_stats["total_passing"],
                    explanation.run_stats["total_failing"],
                    explanation.run_stats["max_depth_reached"],
                    explanation.run_stats["avg_box_size"],
                    multi=multi,
                    multi_no=c,
                )


def add_to_database(
    db,
    args: CausalArgs,
    target,
    confidence,
    responsibility,
    explanation,
    explanation_confidence,
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

    object = DataBaseEntry(
        id,
        args.path,
        target,
        confidence,
        responsibility,
        responsibility_shape,
        explanation,
        explanation_shape,
        explanation_confidence,
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
    object.method = str(args.strategy)
    if args.strategy == Strategy.Spatial:
        object.spatial_radius = args.spatial_initial_radius
        object.spatial_eta = args.spatial_radius_eta
    if args.strategy == Strategy.MultiSpotlight:
        object.spotlights = args.spotlights
        object.spotlight_size = args.spotlight_size
        object.spotlight_eta = args.spotlight_eta
        object.obj_function = args.spotlight_objective_function

    db.add(object)
    db.commit()


class Base(DeclarativeBase):
    pass


class NumpyType(sa.types.TypeDecorator):
    impl = sa.types.LargeBinary

    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            return zlib.compress(value, 9)

    def process_result_value(self, value, dialect):
        return value


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
    explanation_confidence = Column(Float)
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
        explanation_confidence,
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
        spotlights=0,
        spotlight_size=0,
        spotlight_eta=0.0,
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
        self.explanation_confidence = explanation_confidence
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
