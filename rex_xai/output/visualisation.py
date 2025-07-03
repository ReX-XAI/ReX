#!/usr/bin/env python3

"""image generation functions"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch as tt
from PIL import Image, ImageDraw
from matplotlib.figure import Figure
from scipy.ndimage import center_of_mass
from skimage.segmentation import slic
from torch import Tensor

from rex_xai.input.config import CausalArgs
from rex_xai.input.input_data import Data
from rex_xai.responsibility.prediction import Prediction
from rex_xai.utils._utils import add_boundaries, try_detach
from rex_xai.utils.logger import logger


def plot_curve(curve, chunk_size, style="insertion", destination=None):
    # TODO check that this code still works
    """plots an insertion/deletion curve of a responsibility map"""
    curve = np.array(curve)
    x = np.arange(0, len(curve))
    x *= chunk_size

    if style == "insertion":
        plt.plot(x, curve)
        plt.fill_between(x, curve, alpha=0.3)
        area = np.trapz(curve) / len(curve)
        plt.title(f"AUC for normalised insertion curve: {area:5.4f}")
        plt.xlabel("no. pixels")
        plt.ylabel("confidence")
    if style == "deletion":
        curve = 1.0 - curve
        plt.plot(x, curve)
        plt.fill_between(x, curve, alpha=0.3)
        area = np.trapz(curve) / len(curve)
        plt.title(f"AUC for normalised deletion curve: {area:5.4f}")
        plt.xlabel("no. pixels")
        plt.ylabel("confidence")

    if style == "both":
        fig, ax = plt.subplots(1, 2)
        fig.tight_layout()
        ax[0].plot(x, curve)
        ax[0].fill_between(x, curve, alpha=0.3)
        area = np.trapz(curve) / len(curve)
        ax[0].set_xlabel("no pixels")
        ax[0].set_ylabel("confidence")
        ax[0].set_title(f"AUC\ninsertion: {area}")

        curve = 1.0 - curve
        ax[1].plot(x, curve)
        ax[1].fill_between(x, curve, alpha=0.3)
        area = np.trapz(curve) / len(curve)
        ax[1].set_xlabel("no pixels")
        ax[1].set_ylabel("confidence")
        ax[1].set_title(f"AUC\ndeletion: {area}")

        fig.suptitle("Insertion/Deletion Curve")
        plt.subplots_adjust(top=0.85)

    if destination is None:
        plt.show()
    else:
        plt.savefig(destination, bbox_inches="tight", dpi=300, pad_inches=0)


def plot_3d(input, ranking, ogrid, norm=255.0):
    """plots a 3d grid in matplotlib given an image <path>
    If <path> is greyscale or RGBA, it is converted to RGB for plotting.
    """
    # img = Image.open(path).convert("RGB")
    # # TODO this is wrong
    # img = img.resize((ranking.shape[0], ranking.shape[1]))
    img = np.asarray(input)

    img = img / norm  # type: ignore
    if ogrid:
        x, y = np.ogrid[0 : img.shape[0], 0 : img.shape[1]]
    else:
        x, y = np.meshgrid(
            np.arange(0, ranking.shape[0], 1), np.arange(0, ranking.shape[1], 1)
        )
    return img, x, y


# code from https://stackoverflow.com/questions/42481203/heatmap-on-top-of-image
def _transparent_cmap(cmap, N=255):
    "Copy colormap and set alpha values"
    mycmap = cmap
    mycmap._init()
    mycmap._lut[:, -1] = np.linspace(0, 0.8, N + 4)
    return mycmap


def heatmap_plot(data: Data, resp_map, colour, path=None):
    if data.mode == "RGB":
        mycmap = _transparent_cmap(mpl.colormaps[colour])

        y, x = np.mgrid[0 : data.model_height, 0 : data.model_width]
        _, ax = plt.subplots(1, 1)
        if isinstance(data.input, Image.Image):
            input = np.asarray(data.input)
        else:
            input = try_detach(data.input)

        resp_map = try_detach(resp_map)
        ax.imshow(input)
        ax.contourf(x, y, resp_map, 15, cmap=mycmap)
        plt.axis("off")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if path is not None:
            plt.savefig(path, bbox_inches="tight", dpi=300, pad_inches=0)
        else:
            plt.show()


def spectral_plot(explanation, data: Data, ranking, colour, extra=True, path=None):
    if isinstance(ranking, tt.Tensor):
        ranking = ranking.detach().cpu().numpy()
    if isinstance(explanation, tt.Tensor):
        explanation = explanation.detach().cpu().numpy()

    explanation = explanation.squeeze()
    if extra:
        fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
        ax = axs[0]
        axs[1].plot(ranking[0])
        axs[1].set_ylabel("Responsibility")
        axs[1].set_xlabel("Wavenumber")
    else:
        fig, ax = plt.subplots(nrows=1, ncols=1)

    raw_data = data.data[0, 0, :].detach().cpu().numpy()  # type: ignore
    d_min = np.min(raw_data)  # type: ignore
    d_max = np.max(raw_data)  # type: ignore

    # if the spectrum hasn't been base shifted to 0, then we do it to make plotting easier,
    # but we will lie about it on the y axis
    if d_min < 0:
        raw_data += np.abs(d_min)
        y_dmin = np.floor(d_min)
        y_dmax = np.ceil(d_max)
        ytx = np.abs(y_dmin) + y_dmax
        ticks = np.arange(0, ytx)
        labels = [str(x + y_dmin) for x in ticks]
        ax.set_yticks(ticks, labels=labels)
        ranking = np.repeat(ranking, len(labels), axis=0)

    ax.plot(raw_data)  # type: ignore
    ax.set_ylabel("Wave Intensity")

    k = data.target.classification  # type: ignore
    confidence = data.target.confidence  # type: ignore
    fig.suptitle(
        f"Spectrum and Responsibility\nTarget: {k}\nconfidence {confidence:5.4f}"
    )

    mycmap = _transparent_cmap(mpl.colormaps[colour])

    ranking = ranking / np.max(np.abs(ranking))

    c = ax.pcolormesh(ranking, cmap=mycmap)
    if not extra:
        # only plot the colorbar if we are not plotting a separate responsibility plot.
        fig.colorbar(c, ax=ax)

    fig.tight_layout()

    if path is None:
        plt.show()
    else:
        plt.savefig(path, dpi=300)
    plt.close()


def surface_plot(
    input,
    args: CausalArgs,
    resp_map: np.ndarray,
    target: Prediction,
    path=None,
):
    """plots a 3d surface plot"""
    logger.info(
        f"Plotting surface plot for {target.classification} with shape {resp_map.shape} and image shape ({input.height}, {input.width})"
    )
    img, _x, _y = plot_3d(input, resp_map, True)
    fig = plt.figure()

    # TODO enable visualisation of sub-responsibility maps
    keys = [target.classification]

    rows, cols = 1, len(keys)

    # currently this for loop does nothing as keys always has len = 1
    # if passing in multiple responsibility maps, would need to use
    # ranking = resp_maps.get(k) for each iteration
    for i, k in enumerate(keys):
        ranking = resp_map
        if isinstance(ranking, tt.Tensor):
            ranking = ranking.detach().cpu().numpy()
        if ranking is not None:
            ax = fig.add_subplot(rows, cols, i + 1, projection="3d")

            ax.zaxis.set_ticklabels([])  # type: ignore

            ax.plot_surface(  # type: ignore
                _x, _y, np.atleast_2d(0), rstride=5, cstride=5, facecolors=img
            )
            ax.plot_surface(  # type: ignore
                _x,
                _y,
                ranking,
                alpha=0.4,
                cmap=mpl.colormaps[args.heatmap_colours],
            )
            if args.info:
                confidence = target.confidence
                try:
                    x, y = center_of_mass(ranking)
                    x = int(round(x))  # type: ignore
                    y = int(round(y))  # type: ignore
                    z = ranking[x, y]  # type: ignore
                    ax.scatter(x, y, z, color="b")

                    try:
                        lang = os.environ["LANG"]
                    except KeyError:
                        lang = "en_US.UTF-8"
                    if "GB" in lang:
                        ax.text(x, y, z, s="centre of mass")  # type: ignore
                    else:
                        ax.text(x, y, z, s="center of mass")  # type: ignore
                except ValueError:
                    pass

                loc = np.unravel_index(np.argmax(ranking), ranking.shape)
                ax.scatter(loc[0], loc[1], ranking[loc[0], loc[1]], color="r")
                ax.text(loc[0], loc[1], ranking[loc[0], loc[1]], s="max point")  # type: ignore
                if k == target:
                    plt.title(f"Target: {k}\nconfidence {confidence:5.4f}")
                else:
                    plt.title(f"Submap for {k}\nconfidence {confidence:5.4f}")

        if path is None:
            plt.show()
        else:
            plt.savefig(path, bbox_inches="tight", dpi=300)
        plt.close()


def overlay_grid(img, step_count=10):
    draw = ImageDraw.Draw(img)

    y_start = 0
    y_end = img.height
    step_size = int(img.width / step_count)

    for x in range(0, img.width, step_size):
        line = ((x, y_start), (x, y_end))
        draw.line(line, fill=128)

    x_start = 0
    x_end = img.width

    for y in range(0, img.height, step_size):
        line = ((x_start, y), (x_end, y))
        draw.line(line, fill=128)

    del draw

    return img


def remove_background(data: Data, resp_map: np.ndarray) -> np.ndarray:
    """Remove the background from the responsibility map if set in the Data object"""
    if data.mode == "voxel":
        if len(data.data.shape) == 4:
            data_m = data.data[0, :, :, :]
        else:
            data_m = data.data
    else:
        data_m = data.data  # need to check for other modes
    # Set background to minimum value in the responsibility map if set in the Data object
    if data.background is not None and data.background is int or float:
        background = np.where(
            data_m == data.background
        )  # Threshold for background voxels
        resp_map[background] = np.min(resp_map)

    # elif data.background is not None and data.background is tuple:
    #     # Background is a range of values so (x , y) where x is the lower bound and y is the upper bound
    #     background = np.where(
    #         (data_m >= data.background[0]) & (data_m <= data.background[1])
    #     )
    #     resp_map[background] = np.min(resp_map)
    # elif data.background is not None:
    #     logger.warning(
    #         "Background is not set correctly, please check the value. "
    #         "Background value must be an int, float or tuple defining the range of values for the background."
    #     )

    return resp_map


def voxel_plot(args: CausalArgs, resp_map: Tensor, data: Data, path=None):
    """
    Plot a 3D voxel plot of the responsibility map using plotly.
    - Assumes the data is greyscale
    Produces an interactive 3D plot of the data and the responsibility map.
    """
    try:
        import plotly.graph_objs as go
        from dash import Dash, Input, Output, dcc, html
    except ImportError as e:
        logger.error(f"Plotly failed to import caused by {e}.")
        return

    if len(data.data.shape) == 4:
        data_m = data.data[0, :, :, :]
    else:
        data_m = data.data
    resp_map = try_detach(resp_map)  # type: ignore

    # if isinstance(data_m, tt.Tensor):
    #     data_m = data_m.squeeze().detach().cpu().numpy()
    #     tt.tensor(data_m, dtype=tt.float32)
    #
    maps: np.ndarray = resp_map
    resp_map = remove_background(data, maps)

    # Normalize the data
    data_m = (data_m - np.min(data_m)) / (np.max(data_m) - np.min(data_m))
    resp_map = (resp_map - np.min(resp_map)) / (np.max(resp_map) - np.min(resp_map))

    # Check if both data and responsibility map have the same range of values
    assert np.min(data_m) == np.min(resp_map) and np.max(data_m) == np.max(
        resp_map
    ), "Data and Responsibility map must have the same range of values!"

    assert (
        data_m.shape == maps.shape
    ), "Data and Responsibility map must have the same shape!"

    x_max, y_max, z_max = data_m.shape

    colourscales = list(mpl.colormaps.keys())

    app = Dash(__name__)
    x_slice = go.Figure()
    y_slice = go.Figure()
    z_slice = go.Figure()

    app.layout = html.Div(
        [
            html.Div(
                [
                    # X Slice
                    html.Div(
                        [
                            html.Label(
                                "X Slice",
                                style={"font-weight": "bold", "margin-bottom": "10px"},
                            ),
                            dcc.Graph(
                                id="x-slice",
                                style={
                                    "width": "100%",
                                    "height": "auto",
                                    "max-width": "400px",
                                },
                            ),
                            dcc.Slider(
                                0,
                                x_max - 1,
                                1,
                                value=x_max // 2,
                                id="x-slider",
                                marks={0: "0", x_max - 1: str(x_max - 1)},
                                vertical=True,
                                tooltip={"always_visible": True},
                            ),
                        ],
                        style={
                            "display": "flex",
                            "align-items": "center",
                            "gap": "20px",
                            "flex": "1",
                        },
                    ),
                    # Y Slice
                    html.Div(
                        [
                            html.Label(
                                "Y Slice",
                                style={"font-weight": "bold", "margin-bottom": "10px"},
                            ),
                            dcc.Graph(
                                id="y-slice",
                                style={
                                    "width": "100%",
                                    "height": "auto",
                                    "max-width": "400px",
                                },
                            ),
                            dcc.Slider(
                                0,
                                y_max - 1,
                                1,
                                value=y_max // 2,
                                id="y-slider",
                                marks={0: "0", y_max - 1: str(y_max - 1)},
                                vertical=True,
                                tooltip={"always_visible": True},
                            ),
                        ],
                        style={
                            "display": "flex",
                            "align-items": "center",
                            "gap": "20px",
                            "flex": "1",
                        },
                    ),
                ],
                style={"display": "flex", "justify-content": "center", "gap": "40px"},
            ),
            # Z Slice
            html.Div(
                [
                    html.Label(
                        "Z Slice",
                        style={"font-weight": "bold", "margin-bottom": "10px"},
                    ),
                    dcc.Graph(
                        id="z-slice",
                        style={"width": "100%", "height": "auto", "max-width": "400px"},
                    ),
                    dcc.Slider(
                        0,
                        z_max - 1,
                        1,
                        value=z_max // 2,
                        id="z-slider",
                        marks={0: "0", z_max - 1: str(z_max - 1)},
                        vertical=True,
                        tooltip={"always_visible": True},
                    ),
                ],
                style={
                    "display": "flex",
                    "align-items": "center",
                    "gap": "20px",
                    "margin-top": "40px",
                },
            ),
            # Opacity Slider
            html.Div(
                [
                    html.Label("Opacity"),
                    dcc.Slider(
                        0,
                        1,
                        0.1,
                        value=0.5,
                        id="opacity-slider",
                        tooltip={"always_visible": True},
                        marks={0: "0", 1: "1"},
                        vertical=True,
                    ),
                    html.Label("Heatmap Colours"),
                    dcc.Dropdown(
                        id="heatmap-colours",
                        options=colourscales,
                        value=args.heatmap_colours,
                    ),
                ],
                style={
                    "position": "absolute",
                    "top": "10%",
                    "right": "10%",
                    "width": "100px",
                    "outline": "1px solid grey",
                },
            ),
        ],
        style={
            "width": "3000x",
            "height": "20px",
            "margin": "auto",
            "padding": "20px",
            "display": "flex",
            "flex-direction": "column",
            "gap": "40px",
            "outline": "1px solid grey",
        },
    )

    @app.callback(
        Output("x-slice", "figure"),
        Output("y-slice", "figure"),
        Output("z-slice", "figure"),
        Input("x-slider", "value"),
        Input("y-slider", "value"),
        Input("z-slider", "value"),
        Input("opacity-slider", "value"),
        Input("heatmap-colours", "value"),
    )
    def update_slices(
        x_idx, y_idx, z_idx, opacity, heatmap_colours=args.heatmap_colours
    ):
        # X-Slice (YZ plane)
        x_slice.add_trace(
            go.Heatmap(
                z=data_m[x_idx, :, :],
                colorscale="gray_r",
                name="Data",
                zmin=0,
                zmax=1,
                showscale=False,
            )
        )
        x_slice.add_trace(
            go.Heatmap(
                z=resp_map[x_idx, :, :],
                colorscale=heatmap_colours,
                opacity=opacity,
                name="Resp Map",
                zmin=0,
                zmax=1,
            )
        )
        x_slice.update_layout(title=f"YZ Plane at {x_idx}")

        # Y-Slice (XZ plane)
        y_slice.add_trace(
            go.Heatmap(
                z=data_m[:, y_idx, :],
                colorscale="gray_r",
                name="Data",
                zmin=0,
                zmax=1,
                showscale=False,
            )
        )
        y_slice.add_trace(
            go.Heatmap(
                z=resp_map[:, y_idx, :],
                colorscale=heatmap_colours,
                opacity=opacity,
                name="Resp Map",
                zmin=0,
                zmax=1,
            )
        )
        y_slice.update_layout(title=f"XZ Plane at {y_idx}")

        # Z-Slice (XY plane)
        z_slice.add_trace(
            go.Heatmap(
                z=data_m[:, :, z_idx],
                colorscale="gray_r",
                name="Data",
                zmin=0,
                zmax=1,
                showscale=False,
            )
        )
        z_slice.add_trace(
            go.Heatmap(
                z=resp_map[:, :, z_idx],
                colorscale=heatmap_colours,
                opacity=opacity,
                name="Resp Map",
                zmin=0,
                zmax=1,
            )
        )
        z_slice.update_layout(title=f"XY Plane at {z_idx}")

        return x_slice, y_slice, z_slice

    if path is None:
        app.run_server(debug=False, use_reloader=False)
    else:
        # Create a sub figure for each slice and save it for now
        path = path.split(".")[0]
        x_slice.write_image(f"{path}_x_slice.png")
        y_slice.write_image(f"{path}_y_slice.png")
        z_slice.write_image(f"{path}_z_slice.png")


def __transpose_mask(mask: tt.Tensor | np.ndarray, mode: str) -> np.ndarray:
    if mode != "RGB":
        raise TypeError
    mask = try_detach(mask)

    if len(mask.shape) == 4:
        mask = mask.squeeze(0)

    if mask.shape[0] == 3:
        mask = mask.transpose((1, 2, 0))

    return mask


def generate_colours(n, colourmap):
    """
    Generate n evenly spaced RGB colours from a matplotlib colourmap.
    """
    space = np.linspace(0, 1, n)
    colour_space = mpl.colormaps[colourmap].resampled(n)
    # colour space returns colours in RGBA space, so we drop the A at the end
    rgb_colours = [colour_space(i)[:-1] for i in space]
    return rgb_colours


def make_composite_mask(explanations):
    """
    Creates a composite mask from a list of masks.
    """
    composite_mask = None
    for explanation in explanations:
        if composite_mask is None:
            composite_mask = explanation
        else:
            composite_mask = np.where(explanation, 1, composite_mask)

    return composite_mask


def apply_boundaries_to_image(image, explanations, colours):
    """
    Draws the boundaries of the explanations on the image, using the provided colours.
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    for i in range(len(explanations)):
        explanation = explanations[i]

        if explanation.shape[0] == 3:
            explanation = explanation[0, :, :]
        else:
            explanation = explanation[:, :, 0]  # type: ignore

        image = add_boundaries(image, explanation, colour=colours[i])

    return image


def subplot_multi_explanations(image, explanations, titles=None, alpha=0.5):
    if isinstance(image, Image.Image):
        image = np.array(image)

    n = len(explanations)
    cols = min(n, 3)  # max 3 per row
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.array(axes).reshape(-1)

    for idx, (ax, expl) in enumerate(zip(axes, explanations)):
        if expl.ndim == 3:
            # assume (C,H,W) or (H,W,C)
            if expl.shape[0] in (1, 3):
                expl = expl[0]
            else:
                expl = expl[..., 0]

        ax.imshow(image)
        ax.imshow(expl, alpha=alpha)
        if titles:
            ax.set_title(titles[idx])
        ax.axis("off")

    # hide any unused subplots
    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def __save_multi(path, explanations_subset, data, img, colours_subset, args):
    explanations_subset = [
        __transpose_mask(explanation, data.mode) for explanation in explanations_subset
    ]
    composite_mask = make_composite_mask(explanations_subset)

    img = apply_boundaries_to_image(img, explanations_subset, colours_subset)

    if composite_mask is not None:
        cover = np.where(composite_mask, img, args.colour)
        cover = Image.fromarray(cover, data.mode)
        img = Image.fromarray(img, data.mode)
        out = Image.blend(cover, img, args.alpha)

        if path is None:
            return out
        else:
            out.save(path)


def save_contrastive(explanation, data, args: CausalArgs, path=None):
    colours_subset = [0.222, 1.0]

    explanations_subset = []
    explanations_subset.append(__transpose_mask(explanation.sufficiency_mask, "RGB"))
    explanations_subset.append(__transpose_mask(explanation.necessity_mask, "RGB"))

    composite_mask = make_composite_mask(explanations_subset)

    img = apply_boundaries_to_image(data.input, explanations_subset, colours_subset)

    if composite_mask is not None:
        cover = np.where(composite_mask, img, args.colour)
        cover = Image.fromarray(cover, data.mode)
        img = Image.fromarray(img, data.mode)
        out = Image.blend(cover, img, args.alpha)

        if path is None:
            return out
        else:
            out.save(path)


def save_complete(explanation, data, args: CausalArgs, path=None):
    colours_subset = [0.222, 0.5, 1.0]

    explanations_subset = []
    explanations_subset.append(__transpose_mask(explanation.sufficiency_mask, "RGB"))
    explanations_subset.append(__transpose_mask(explanation.necessity_mask, "RGB"))
    explanations_subset.append(__transpose_mask(explanation.complete_mask, "RGB"))

    composite_mask = make_composite_mask(explanations_subset)

    img = apply_boundaries_to_image(data.input, explanations_subset, colours_subset)

    if composite_mask is not None:
        cover = np.where(composite_mask, img, args.colour)
        cover = Image.fromarray(cover, data.mode)
        img = Image.fromarray(img, data.mode)
        out = Image.blend(cover, img, args.alpha)

        if path is None:
            return out
        else:
            out.save(path)


def save_multi_explanation(
    explanations, data, args: CausalArgs, clause=None, path=None
):
    if data.mode != "RGB":
        logger.warning("we do not yet handle multiple explanations for non-images")
        raise NotImplementedError

    img = data.input

    if img is not None:
        rgb_colours = generate_colours(args.spotlights, args.heatmap_colours)

        if clause is not None:
            if isinstance(clause, int):
                explanations_subset = [explanations[clause]]
                colours_subset = [rgb_colours[clause]]
                __save_multi(path, explanations_subset, data, img, colours_subset, args)
            else:
                explanations_subset = [explanations[c] for c in clause]
                colours_subset = [rgb_colours[c] for c in clause]
                __save_multi(path, explanations_subset, data, img, colours_subset, args)


def save_image(mask: tt.Tensor | np.ndarray, data: Data, args: CausalArgs, path=None):
    if data.mode == "RGB":
        if len(data.input.size) == 4:
            data.input = data.input.squeeze(0)
        img = data.input

        mask = __transpose_mask(mask, data.mode)

        if args.raw:
            out = np.where(mask, img, 0).squeeze(0)  # 0 used to mask image with black
            out = Image.fromarray(out, data.mode)
        elif args.mask_value == "context":
            # Use preprocessed context and data as shape needs to match
            if isinstance(data.context, tt.Tensor):
                data.context = data.context.squeeze().detach().cpu().numpy()
            if isinstance(data.data, tt.Tensor):
                data.data = data.data.squeeze().detach().cpu().numpy()
            context = __transpose_mask(data.context, data.mode)
            img = __transpose_mask(data.data, data.mode)
            fig = np.where(mask == False, context, img)
            out, ax = plt.subplots()
            ax.imshow(fig)
            ax.axis("off")
        else:
            exp = np.where(mask, img, args.colour)
            exp = Image.fromarray(exp, "RGB")
            out = Image.blend(exp, img, args.alpha)

            if args.mark_segments:
                segs = slic(np.array(img))
                m = add_boundaries(np.array(img), segs)
                marked = Image.fromarray(m, data.mode)
                out = Image.blend(out, marked, args.alpha)

            if args.grid:
                out = overlay_grid(out)

        if path is not None:
            if isinstance(out, Figure):
                out.savefig(path, dpi=300, pad_inches=0)
            else:
                out.save(path)
            logger.info(f"Saved explanation to {path}")

        return out

    elif data.mode == "voxel":
        data_m: np.ndarray = data.data  # type:ignore
        if isinstance(mask, tt.Tensor):
            mask = mask.squeeze().detach().cpu().numpy()
        else:
            mask = mask.squeeze()
        data_m = data_m[0, :, :, :]  # Remove batch dimension

        mask = remove_background(data, mask)

        num_slices = 10
        _, axes = plt.subplots(3, num_slices, figsize=(15, 6))

        for axis, _ in enumerate(mask.shape):
            slice_indices = np.linspace(0, axis - 1, num_slices, dtype=int)
            for i, slice_index in enumerate(slice_indices):
                ax = axes[axis, i]
                data_slice = np.take(data_m, slice_index, axis=axis)
                resp_slice = np.take(mask, slice_index, axis=axis)
                ax.imshow(data_slice, cmap="gray")
                ax.imshow(resp_slice, cmap=args.heatmap_colours, alpha=0.4)
                ax.axis("off")

        plt.tight_layout()

        if args.output is not None:
            plt.savefig(args.output)
        else:
            plt.show()


def plot_image_grid(images, ncols=None):
    # adapted from: https://stackoverflow.com/questions/41793931/plotting-images-side-by-side-using-matplotlib/66961099#66961099
    """Plot a grid of images"""
    if not ncols:
        factors = [i for i in range(1, len(images) + 1) if len(images) % i == 0]
        ncols = factors[len(factors) // 2] if len(factors) else len(images) // 4 + 1
    nrows = int(len(images) / ncols) + int(len(images) % ncols)
    imgs = [images[i] if len(images) > i else None for i in range(nrows * ncols)]
    f, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2 * nrows))
    axes = axes.flatten()[: len(imgs)]
    for img, ax in zip(imgs, axes.flatten()):
        ax.imshow(img)
        ax.set_axis_off()
    f.tight_layout()
