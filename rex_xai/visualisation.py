#!/usr/bin/env python3

"""image generation functions"""

from PIL import Image, ImageDraw
import os

import torch as tt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.ndimage import center_of_mass
from skimage.segmentation import slic
from torch import Tensor

from rex_xai.prediction import Prediction
from rex_xai.config import CausalArgs
from rex_xai.resp_maps import ResponsibilityMaps
from rex_xai.input_data import Data
from rex_xai._utils import add_boundaries
from rex_xai.logger import logger


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


def plot_3d(path, ranking, ogrid, norm=255.0):
    """plots a 3d grid in matplotlib given an image <path>
    If <path> is greyscale or RGBA, it is converted to RGB for plotting.
    """
    img = Image.open(path).convert("RGB")
    img = img.resize((ranking.shape[0], ranking.shape[1]))
    img = np.asarray(img)

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
    if data.mode in ("RGB", "L"):
        mycmap = _transparent_cmap(mpl.colormaps[colour])
        background = data.input.resize(
            (data.model_height, data.model_width)
        )  # TODO check these dimensions
        y, x = np.mgrid[0 : data.model_height, 0 : data.model_width]
        fig, ax = plt.subplots(1, 1)
        ax.imshow(background)
        ax.contourf(x, y, resp_map, 15, cmap=mycmap)
        plt.axis("off")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if path is not None:
            plt.savefig(path, bbox_inches="tight", dpi=300, pad_inches=0)
        else:
            plt.show()


def contour_plot(path, maps: ResponsibilityMaps, target, levels=30, destination=None):
    """plots a contour plot"""
    pass


def __group_spectral_parts(explanation):
    # coords = tt.where(explanation)[0]

    coords = np.where(explanation.detach().cpu().numpy())[0]

    res = []
    local = [coords[0]]
    p = 1
    while p < len(coords):
        if coords[p] == coords[p - 1] + 1:
            local.append(coords[p])
        else:
            res.append(local)
            local = [coords[p]]
        p += 1
    res.append(local)

    return res


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
        # only plot the colorbar is we are not plotting a separate responsibility plot.
        fig.colorbar(c, ax=ax)

    fig.tight_layout()

    # coords = __group_spectral_parts(explanation)
    #
    # for rect in coords:
    #     rectangle = Rectangle(
    #         (rect[0], 0), rect[-1] - rect[0], 3, alpha=0.3, color="red"
    #     )
    #     axs[0].add_patch(rectangle)

    if path is None:
        plt.show()
    else:
        plt.savefig(path, dpi=300)


def surface_plot(
    args: CausalArgs,
    resp_map: np.ndarray,
    target: Prediction,
    path=None,
):
    """plots a 3d surface plot"""
    img, _x, _y = plot_3d(args.path, resp_map, True)
    fig = plt.figure()

    # TODO enable visualisation of sub-responsibility maps
    keys = [target.classification]

    rows, cols = 1, len(keys)

    # currently this for loop does nothing as keys always has len = 1
    # if passing in multiple responsibility maps, would need to use
    # ranking = resp_maps.get(k) for each iteration
    for i, k in enumerate(keys):
        ranking = resp_map
        if ranking is not None:
            ax = fig.add_subplot(rows, cols, i + 1, projection="3d")

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
                # confidence = 0.0
                # if k == target.classification:
                confidence = target.confidence
                # else:
                #     for p in args.extra_targets:
                #         if p.pred == k:
                #             confidence = p.conf
                #             break

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
        data_m = data.data # need to check for other modes
    # Set background to minimum value in the responsibility map if set in the Data object
    if data.background is not None and data.background is int or float:
        background = np.where(
            data_m == data.background
        )  # Threshold for background voxels
        resp_map[background] = np.min(resp_map)
    elif data.background is not None and data.background is tuple:
        # Background is a range of values so (x , y) where x is the lower bound and y is the upper bound
        background = np.where(
            (data_m >= data.background[0]) & (data_m <= data.background[1])
        )
        resp_map[background] = np.min(resp_map)
    elif data.background is not None:
        logger.warn(
            "Background is not set correctly, please check the value. "
            "Background value must be an int, float or tuple defining the range of values for the background."
        )

    return resp_map


def voxel_plot(args: CausalArgs, resp_map: Tensor, data: Data, path=None):
    """
    Plot a 3D voxel plot of the responsibility map using plotly.
    - Assumes the data is greyscale
    Produces an interactive 3D plot of the data and the responsibility map.
    """
    try:
        import plotly.graph_objs as go
        from dash import Dash, dcc, html, Input, Output
    except ImportError as e:
        logger.error(f"Plotly failed to import caused by {e}.")
        return

    if len(data.data.shape) == 4:
        data_m = data.data[0, :, :, :]
    else:
        data_m = data.data
    if isinstance(resp_map, tt.Tensor):
        resp_map = resp_map.squeeze().detach().cpu().numpy()

    if isinstance(data_m, tt.Tensor):
        data_m = data_m.squeeze().detach().cpu().numpy()
        tt.tensor(data_m, dtype=tt.float32)

    maps: np.ndarray = resp_map
    resp_map = remove_background(data, maps)

    # Normalize the data
    data_m = (data_m - np.min(data_m)) / (np.max(data_m) - np.min(data_m))
    resp_map = (resp_map - np.min(resp_map)) / (np.max(resp_map) - np.min(resp_map))

    # Check if both data and responsibility map have the same range of values
    assert np.min(data_m) == np.min(resp_map) and np.max(data_m) == np.max(resp_map), (
        "Data and Responsibility map must have the same range of values!"
    )

    assert data_m.shape == maps.shape, (
        "Data and Responsibility map must have the same shape!"
    )

    x_max, y_max, z_max = data_m.shape

    colourscales = list(mpl.colormaps.keys())

    app = Dash(__name__)
    x_slice = go.Figure()
    y_slice = go.Figure()
    z_slice = go.Figure()

    app.layout = html.Div([
        html.Div([
            # X Slice
            html.Div([
                html.Label("X Slice", style={"font-weight": "bold", "margin-bottom": "10px"}),
                dcc.Graph(id="x-slice", style={"width": "100%", "height": "auto", "max-width": "400px"}),
                dcc.Slider(0, x_max - 1, 1, value=x_max // 2, id="x-slider",
                           marks={0: "0", x_max - 1: str(x_max - 1)},
                           vertical=True, tooltip={"always_visible": True}),
            ], style={"display": "flex", "align-items": "center", "gap": "20px", "flex": "1"}),

            # Y Slice
            html.Div([
                html.Label("Y Slice", style={"font-weight": "bold", "margin-bottom": "10px"}),
                dcc.Graph(id="y-slice", style={"width": "100%", "height": "auto", "max-width": "400px"}),
                dcc.Slider(0, y_max - 1, 1, value=y_max // 2, id="y-slider",
                           marks={0: "0", y_max - 1: str(y_max - 1)},
                           vertical=True, tooltip={"always_visible": True}),
            ], style={"display": "flex", "align-items": "center", "gap": "20px", "flex": "1"}),

        ], style={"display": "flex", "justify-content": "center", "gap": "40px"}),

        # Z Slice
        html.Div([
            html.Label("Z Slice", style={"font-weight": "bold", "margin-bottom": "10px"}),
            dcc.Graph(id="z-slice", style={"width": "100%", "height": "auto", "max-width": "400px"}),
            dcc.Slider(0, z_max - 1, 1, value=z_max // 2, id="z-slider",
                       marks={0: "0", z_max - 1: str(z_max - 1)},
                       vertical=True, tooltip={"always_visible": True}),
        ], style={"display": "flex", "align-items": "center", "gap": "20px", "margin-top": "40px"}),

        # Opacity Slider
        html.Div([
            html.Label("Opacity"),
            dcc.Slider(0, 1, 0.1, value=0.5, id="opacity-slider",
                       tooltip={"always_visible": True}, marks={0: "0", 1: "1"},
                       vertical=True),
            html.Label("Heatmap Colours"),
            dcc.Dropdown(id="heatmap-colours", options=colourscales, value=args.heatmap_colours)
        ], style={"position": "absolute", "top": "10%", "right": "10%", "width": "100px", "outline": "1px solid grey"}),

    ], style={"width": "3000x", "height": "20px", "margin": "auto", "padding": "20px", "display": "flex",
              "flex-direction": "column", "gap": "40px", "outline": "1px solid grey"})

    @app.callback(
        Output("x-slice", "figure"),
        Output("y-slice", "figure"),
        Output("z-slice", "figure"),
        Input("x-slider", "value"),
        Input("y-slider", "value"),
        Input("z-slider", "value"),
        Input("opacity-slider", "value"),
        Input("heatmap-colours", "value")
    )
    def update_slices(x_idx, y_idx, z_idx, opacity, heatmap_colours=args.heatmap_colours):
        # X-Slice (YZ plane)
        x_slice.add_trace(
            go.Heatmap(z=data_m[x_idx, :, :], colorscale="gray_r", name="Data", zmin=0, zmax=1, showscale=False))
        x_slice.add_trace(
            go.Heatmap(z=resp_map[x_idx, :, :], colorscale=heatmap_colours, opacity=opacity, name="Resp Map", zmin=0, zmax=1))
        x_slice.update_layout(title=f"YZ Plane at {x_idx}")

        # Y-Slice (XZ plane)
        y_slice.add_trace(
            go.Heatmap(z=data_m[:, y_idx, :], colorscale="gray_r", name="Data", zmin=0, zmax=1, showscale=False))
        y_slice.add_trace(
            go.Heatmap(z=resp_map[:, y_idx, :], colorscale=heatmap_colours, opacity=opacity, name="Resp Map", zmin=0, zmax=1))
        y_slice.update_layout(title=f"XZ Plane at {y_idx}")

        # Z-Slice (XY plane)
        z_slice.add_trace(
            go.Heatmap(z=data_m[:, :, z_idx], colorscale="gray_r", name="Data", zmin=0, zmax=1, showscale=False))
        z_slice.add_trace(
            go.Heatmap(z=resp_map[:, :, z_idx], colorscale=heatmap_colours, opacity=opacity, name="Resp Map", zmin=0, zmax=1))
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



def __transpose_mask(explanation, mode, transposed):
    mask = None
    if transposed:
        if mode == "RGB":
            mask = explanation.squeeze().detach().cpu().numpy().transpose((1, 2, 0))
        elif mode == "L":
            mask = explanation.squeeze(0).detach().cpu().numpy().transpose((2, 1, 0))
            mask = np.repeat(mask, 3, axis=-1)
    else:
        mask = explanation.squeeze(0).detach().cpu().numpy()

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
    for i in range(len(explanations)):
        explanation = explanations[i]

        if explanation.shape[0] == 3:
            explanation = explanation[0, :, :]
        else:
            explanation = explanation[:, :, 0]  # type: ignore

        image = add_boundaries(image, explanation, colour=colours[i])

    return image


def get_img_as_array(data):
    """
    Return original input image as a numpy array, resized to match model size and converted to RGB if necessary.
    """
    if data.mode == "RGB" or data.mode == "L":
        if data.mode == "L":
            img = data.input.convert("RGB").resize(
                (data.model_height, data.model_width)
            )
        else:
            img = data.input.resize((data.model_height, data.model_width))
    else:
        raise NotImplementedError

    return np.array(img)


def save_multi_explanation(
    explanations, data, args: CausalArgs, clause=None, path=None
):
    if data.mode == "RGB" or data.mode == "L":
        img = get_img_as_array(data)
    else:
        logger.warning("we do not yet handle multiple explanations for non-images")
        raise NotImplementedError

    if img is not None:
        rgb_colours = generate_colours(args.spotlights, args.heatmap_colours)

        if clause is not None:
            if isinstance(clause, int):
                explanations_subset = [explanations[clause]]
                colours_subset = [rgb_colours[clause]]
            else:
                explanations_subset = [explanations[c] for c in clause]
                colours_subset = [rgb_colours[c] for c in clause]
            explanations_subset = [
                __transpose_mask(explanation, data.mode, data.transposed)
                for explanation in explanations_subset
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


def save_image(explanation, data: Data, args: CausalArgs, path=None):
    mask = None
    if data.mode == "RGB" or data.mode == "L":
        if data.mode == "L":
            img = data.input.convert("RGB").resize(
                (data.model_height, data.model_width)
            )
        else:
            img = data.input.resize((data.model_height, data.model_width))

        mask = __transpose_mask(explanation, data.mode, data.transposed)

        if mask is not None:
            if args.raw:
                out = np.where(mask, img, 0).squeeze(
                    0
                )  # 0 used to mask image with black
                out = Image.fromarray(out, data.mode)

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

                if args.resize:
                    out = out.resize(data.input.size)

            if path is not None:
                out.save(path)

            return out
    elif data.mode == "voxel":
        data_m: np.ndarray = data.data
        if isinstance(explanation, tt.Tensor):
            explanation = explanation.squeeze().detach().cpu().numpy()
        else:
            explanation = explanation.squeeze()
        data_m = data_m[0, :, :, :]  # Remove batch dimension

        explanation = remove_background(data, explanation)

        num_slices = 10
        fig, axes = plt.subplots(3, num_slices, figsize=(15, 6))

        for axis, index in enumerate(explanation.shape):
            slice_indices = np.linspace(0, axis - 1, num_slices, dtype=int)
            for i, slice_index in enumerate(slice_indices):
                ax = axes[axis, i]
                data_slice = np.take(data_m, slice_index, axis=axis)
                resp_slice = np.take(explanation, slice_index, axis=axis)
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
