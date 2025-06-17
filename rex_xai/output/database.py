#!/usr/bin/env python
from __future__ import annotations

import zlib
from ast import literal_eval
from datetime import datetime

import numpy as np
import pandas as pd
import sqlalchemy as sa
import torch as tt
from sqlalchemy import Boolean, Column, Float, Integer, String, Unicode, create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from rex_xai.explanation.explanation import Explanation
from rex_xai.explanation.multi_explanation import MultiExplanation
from rex_xai.input.config import CausalArgs, Strategy
from rex_xai.utils._utils import try_detach
from rex_xai.utils.logger import logger


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


# def __multi_update(
#     db,
#     explanation,
#     classification,
#     target,
#     target_map,
#     sufficiency_mask,
#     time_taken,
#     multi_no,
# ):
#     if isinstance(sufficiency_mask, tt.Tensor):
#         sufficiency_mask = sufficiency_mask.detach().cpu().numpy()
#     pass
# add_to_database(
#     db,
#     explanation.args,
#     classification,
#     target.confidence,
#     target_map,
#     final_mask,
#     explanation.explanation_confidences[multi_no],
#     time_taken,
#     explanation.run_stats["total_passing"],
#     explanation.run_stats["total_failing"],
#     explanation.run_stats["max_depth_reached"],
#     explanation.run_stats["avg_box_size"],
#     multi=True,
#     multi_no=multi_no,
# )


def update_database(
    db,
    explanation: Explanation | MultiExplanation,  # type: ignore
    time_taken: float,
    multi=False,
    clauses=None,
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
        if explanation.sufficiency_mask is None:
            logger.warning("unable to update database as explanation is empty")
            return
        sufficiency_mask = try_detach(explanation.sufficiency_mask)

        necessity_mask = None
        complete_mask = None
        inverse_classification = None
        inverse_confidence = None
        necessity_confidence = None
        complete_classification = None
        complete_confidence = None
        if hasattr(explanation, "necessity_mask"):
            necessity_mask = try_detach(explanation.necessity_mask)
            necessity_confidence = explanation.necessity_confidence  # type: ignore
            inverse_classification = explanation.inverse_classification
            inverse_confidence = explanation.inverse_confidence
        if hasattr(explanation, "complete_mask"):
            complete_mask = try_detach(explanation.complete_mask)
            complete_confidence = explanation.completeness_confidence
            complete_classification = explanation.completeness_classification

        explanation_confidence = explanation.sufficiency_confidence

        add_to_database(
            db,
            explanation.args,
            classification,
            target.confidence,
            target_map,
            sufficiency_mask,
            explanation_confidence,
            necessity_mask,
            necessity_confidence,
            inverse_classification,
            inverse_confidence,
            complete_mask,
            complete_classification,
            complete_confidence,
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
            for c, sufficiency_mask in enumerate(explanation.explanations):
                if clauses is not None:
                    if c not in clauses:
                        logger.warning("ignoring %s", c)
                    pass
                #     else:
                #         __multi_update(
                #             db,
                #             explanation,
                #             classification,
                #             target,
                #             target_map,
                #             final_mask,
                #             time_taken,
                #             c,
                #         )
                # else:
                #     __multi_update(
                #         db,
                #         explanation,
                #         classification,
                #         target,
                #         target_map,
                #         final_mask,
                #         time_taken,
                #         c,
                #     )
                #


def add_to_database(
    db,
    args: CausalArgs,
    target: int,
    confidence: float | None,
    responsibility,
    sufficiency_mask,
    sufficiency_confidence: float | None,
    contrastive_mask,
    contrastive_confidence: float | None,
    inverse_classification: int | None,
    inverse_confidence: float | None,
    complete_mask,
    complete_classification: int | None,
    complete_confidence,
    time_taken: float,
    passing: int,
    failing: int,
    depth_reached: int,
    avg_box_size: float,
    multi=False,
    multi_no=None,
):
    if multi:
        id = hash(str(datetime.now().time()) + str(multi_no))
    else:
        id = hash(str(datetime.now().time()))

    responsibility_shape = responsibility.shape
    explanation_shape = sufficiency_mask.shape

    object = DataBaseEntry(
        id,
        args.path,
        target,
        confidence,
        responsibility,
        responsibility_shape,
        sufficiency_mask,
        explanation_shape,
        sufficiency_confidence,
        time_taken,
        passing=passing,
        failing=failing,
        contrastive_mask=contrastive_mask,
        contrastive_confidence=contrastive_confidence,
        complete_mask=complete_mask,
        complete_classification=complete_classification,
        complete_confidence=complete_confidence,
        inverse_classification=inverse_classification,
        inverse_confidence=inverse_confidence,
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
    # basic explanation type
    sufficiency_mask = Column(NumpyType)
    mask_shape = Column(Unicode)
    sufficiency_confidence = Column(Float)
    # sufficient and necessary mask
    contrastive_mask = Column(NumpyType)
    contrastive_confidence = Column(Float)

    inverse_classification = Column(Integer)
    inverse_confidence = Column(Float)

    # complete mask
    complete_mask = Column(NumpyType)
    complete_classification = Column(Integer)
    complete_confidence = Column(Float)

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
        sufficiency_mask,
        mask_shape,
        sufficiency_confidence,
        time_taken,
        contrastive_mask=None,
        contrastive_confidence=None,
        inverse_classification=None,
        inverse_confidence=None,
        complete_mask=None,
        complete_classification=None,
        complete_confidence=None,
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
        self.sufficiency_mask = sufficiency_mask
        self.mask_shape = str(mask_shape)
        self.sufficiency_confidence = sufficiency_confidence
        self.time = time_taken
        self.total_work = total_work
        self.passing = passing
        self.failing = failing

        # contrastive and complete explanations
        self.contrastive_mask = contrastive_mask
        self.contrastive_confidence = contrastive_confidence
        self.inverse_classification = inverse_classification
        self.inverse_confidence = inverse_confidence
        self.complete_mask = complete_mask
        self.complete_classification = complete_classification
        self.complete_confidence = complete_confidence

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
