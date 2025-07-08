#!/usr/bin/env python

"""
calculate causal responsibility
"""

from collections import deque
from typing import List

import numpy as np
import torch as tt

try:
    from anytree.cachedsearch import find
except ImportError:
    from anytree.search import find


from rex_xai.input.config import CausalArgs, Queue
from rex_xai.input.input_data import Data
from rex_xai.mutants.box import average_box_size, initialise_tree
from rex_xai.mutants.mutant import Mutant, _apply_to_data, get_combinations
from rex_xai.responsibility.prediction import Prediction
from rex_xai.responsibility.resp_maps import ResponsibilityMaps
from rex_xai.utils.logger import logger


def subbox(tree, name, max_depth, min_size, mode, r_map=None):
    """Split a box into 4 contiguous children or None if no possible.

    @param tree: a search tree
    @param name: a node name to search for in <tree>
    @param max_depth: tree depth limit for new children
    @param min_size: minimum new child size
    @param mode: spectral, tabular, RGB, L or voxel
    @param r_map=None: responsibility map

    @return None or new children
    """
    active = find(tree, lambda node: node.name == name)
    if active is not None:
        if active.depth <= max_depth and len(active.children) == 0:
            active.add_children_to_tree(min_size, mode, r_map)
        return active.children
    return None


def update_depth_reached(depth_reached, passing):
    """Update the max tree depth reached.

    @param depth_reached: current max_depth
    @param passing: a list of passing Mutant

    @return int
    """
    mp = [m.depth for m in passing]
    if mp == []:
        mp = 0
    else:
        mp = max(mp)
    return max(depth_reached, mp)


def prune(mutants: List[Mutant], technique=Queue.Intersection, keep=None):
    """Reduce the search queue to improve both efficiency and (possibly) result quality.

    @param mutants: a list of passing Mutant objects
    @param technique: a Queue enum
    @param keep=None: how many items to keep in the queue, all if keep is None

    @return a list of mutants of length <= keep
    """
    # we use "none" when we are looking for multiple explanations. It has a tendency
    # to produce flatter landscapes than intersection
    if technique == Queue.All:
        return mutants
    if technique == Queue.Intersection and len(mutants) > 1:
        inter = set()
        i = 0
        while len(inter) == 0 and i < len(mutants):
            names = [mutant.get_active_boxes() for mutant in mutants]
            head = set(names[i])
            tail = names[i + 1 :]
            inter = head.intersection(*tail)
            i += 1
        mutants = [m for m in mutants if inter <= set(m.get_active_boxes())]
        ordered = sorted(mutants, key=lambda x: x.area())
        if keep is not None:
            return ordered[:keep]
        return ordered

    if technique == Queue.Area:
        ordered = sorted(mutants, key=lambda x: x.area())
        if keep is not None:
            return ordered[:keep]
        return ordered

    return mutants


def causal_explanation(
    process, data: Data, args: CausalArgs, prediction_func, current_map=None
):
    """Calculate causal responsiblity.

    @param process: an integer value
    @param data: a Data object
    @param args: a CausalArgs object
    @param prediction_func: a higher order
        function that calls a model and return a Prediction object
    """

    assert data.target is not None

    if args.seed is not None:
        np.random.seed(args.seed + process)
        tt.manual_seed(args.seed + process)

    if args.mask_value in ("random", "linear"):
        lower = tt.min(data.data).item()  # type: ignore
        upper = tt.max(data.data).item()  # type: ignore

        if args.mask_value == "random":
            data.mask_value = np.random.uniform(lower, upper)
        else:
            steps = np.linspace(lower, upper, args.iters)
            data.mask_value = steps[process - 1]  # type: ignore
        logger.info("using %.3f for process %d", data.mask_value, process)

    if args.use_bounding_box:
        assert data.target.bounding_box is not None
        logger.info(
            f"Using bounding box bounding box for {data.target.classification} that has the bounding box {data.target.bounding_box}"
        )
        box = data.target.bounding_box
        search_tree = initialise_tree(
            int(box[3]),
            int(box[2]),
            args.distribution,
            args.distribution_args,
            d_lim=data.model_depth,
            r_start=int(box[1]),
            c_start=int(box[0]),
        )
    else:
        search_tree = initialise_tree(
            data.model_height,
            data.model_width,
            args.distribution,
            args.distribution_args,
            d_lim=data.model_depth,
        )

    total_work = 0
    total_passing = 0
    total_failing = 0

    depth_reached = 0

    # The <queue> is a list of strings in the form "R:x:y:...n"
    queue = deque(search_tree.name)

    local_maps = ResponsibilityMaps(args.responsibility_style)

    # a <job> is of the form "R:x:y:...n", where x,y...n are integers.
    # This is both the unique name for a passing mutant and the node name for
    # the node in <search_tree>
    while True:
        passing = []
        while len(queue) != 0:
            job = queue.popleft()
            sub_jobs = job.split("_")

            todo = len(sub_jobs) - 1
            for ai, active in enumerate(sub_jobs):
                static = [sj for x, sj in enumerate(sub_jobs) if x != ai]

                child_boxes = subbox(
                    search_tree,
                    active,
                    args.tree_depth,
                    args.min_box_size,
                    data.mode,
                    r_map=current_map,
                )

                if child_boxes is None or len(child_boxes) == 0:
                    logger.debug("no children, breaking")
                    break

                mutants = np.empty(14, dtype=np.object_)
                if child_boxes is not None:
                    for j, combination in enumerate(get_combinations()):
                        nps = [child_boxes[i] for i in combination]
                        current = "_".join([b.name for b in nps])

                        m = Mutant(
                            data,
                            static=static,
                            active=current,
                            masking_func=data.mask_value,
                        )
                        m.set_active_mask_regions(nps)
                        m.set_static_mask_regions(static, search_tree)
                        mutants[j] = m

                work_done = len(mutants)

                # TODO find out why this was added
                def apply_mask(m):
                    return _apply_to_data(m.mask, data)

                if data.mode in ("spectral", "tabular"):
                    preds: List[Prediction] = [
                        prediction_func(apply_mask(m))[0] for m in mutants
                    ]
                else:
                    # TODO this needs testing
                    if args.batch_size == 1:
                        preds = [
                            prediction_func(
                                apply_mask(m),  #  type: ignore
                                data.target,
                            )[0]
                            for m in mutants
                        ]  # type: ignore
                    else:
                        tensors = tt.stack(
                            [
                                apply_mask(m)  #  type: ignore
                                for m in mutants
                            ]
                        )  # type: ignore
                        if len(tensors.shape) > len(data.model_shape):
                            tensors = tensors.squeeze(1)
                        preds: List[Prediction] = prediction_func(
                            tensors,
                            data.target,
                        )

                for i, m in enumerate(mutants):
                    m.prediction = preds[i]
                    m.update_status(data.target)

                passing: List[Mutant] = list(
                    filter(
                        lambda m: m.passing
                        and m.prediction.confidence
                        >= (data.target.confidence * args.confidence_filter),  # type: ignore
                        mutants,
                    )
                )

                if args.verbosity > 3:
                    n = 0
                    for m in mutants:
                        m.save_mutant(
                            data,
                            f"{process}_{m.depth}_{n}_{m.prediction.confidence}_{m.passing}.png",
                        )
                        n += 1

                total_passing += len(passing)
                total_failing += work_done - len(passing)
                total_work += work_done

                # we have no passing occlusions
                if not passing:
                    if ai == todo:
                        logger.debug(
                            "there are no passing mutants at %d, so quitting here",
                            depth_reached,
                        )
                        # logger.debug(global_queue)
                        break

                # something passed...
                else:
                    # update responsibilities
                    local_maps.update_maps(mutants, args, data, search_tree)  # type: ignore

                    # reduce the elements to add to the search queue
                    passing = prune(
                        passing, technique=args.queue_style, keep=args.queue_len
                    )  # type: ignore

                depth_reached = update_depth_reached(depth_reached, passing)

        # if we are too deep into the tree, break from the loop
        if depth_reached > args.tree_depth and ai == todo:  # type: ignore
            logger.info("breaking at %s as max tree depth reached", depth_reached)
            break

        if args.search_limit is not None and total_work > args.search_limit:
            logger.info("exceeded total work limit for this iteration")
            break

        if args.queue_style == Queue.DC:
            update = list(set([m.get_name() for m in passing] + list(queue)))
        else:
            update = [m.get_name() for m in passing]
        if update == []:
            logger.debug("nothing left in the queue")
            break

        # our new search queue, which takes us back to the beginning
        queue = deque(update)

    # clear up unneeded mutants and boxes
    if data.device == "mps":
        with tt.no_grad():
            tt.mps.empty_cache()
    elif data.device == "cuda":
        with tt.no_grad():
            tt.cuda.empty_cache()

    logger.info(
        "total work %d with %d passing and %d failing, max depth explored %d",
        total_work,
        total_passing,
        total_failing,
        depth_reached,
    )

    return (
        local_maps,
        total_passing,
        total_failing,
        depth_reached,
        average_box_size(search_tree, depth_reached),
    )
