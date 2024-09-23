#!/usr/bin/env python3

"""image generation functions"""

from PIL import Image, ImageDraw
import os

# from matplotlib.text import Rectangle #type: ignore
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from scipy.ndimage import center_of_mass
from skimage.segmentation import slic

from rex_ai.prediction import Prediction
from rex_ai.config import CausalArgs
from rex_ai.resp_maps import ResponsibilityMaps
from rex_ai.input_data import Data
from rex_ai._utils import add_boundaries


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


def plot_3d(path, ranking, ogrid):
    """plots a 3d grid in matplotlib"""
    img = Image.open(path)
    img = img.resize((ranking.shape[0], ranking.shape[1]))
    img = np.asarray(img)

    img = img / 255.0  # type: ignore
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
    mycmap._lut[:,-1] = np.linspace(0, 0.8, N+4)
    return mycmap


def heatmap_plot(data: Data, resp_maps, colour, target: Prediction, path=None):
    if data.mode in ("RGB", "L"):
        mycmap = _transparent_cmap(mpl.colormaps[colour])
        background = data.input.resize((data.model_height, data.model_width)) # TODO check these dimensions
        x, y = np.mgrid[0:data.model_height, 0:data.model_width]
        fig, ax = plt.subplots(1, 1)
        ax.imshow(background)
        resp_map = resp_maps.get(target.classification)
        # TODO this is not working as expected
        resp_map = np.rot90(resp_map, k=2)
        ax.contourf(x, y, resp_map, 15, cmap=mycmap)
        plt.axis('off')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if path is not None:
            plt.savefig(path, bbox_inches='tight', dpi=300, pad_inches=0)
        else:
            plt.show()



def contour_plot(
    path, maps: ResponsibilityMaps, target, levels=30, destination=None
):
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


def spectral_plot(args: CausalArgs, explanation, data: Data, ranking, colour, extra=False):
    # plt.style.use("seaborn-v0_8")
    explanation = explanation.squeeze()
    if extra:
        fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
        ax = axs[0]
        axs[1].plot(ranking[0])
        axs[1].set_ylabel("Responsibility")
        axs[1].set_xlabel("Wavenumber")
    else:
        fig, ax = plt.subplots(nrows=1, ncols=1)


    data = data.data[0, 0, :].detach().cpu().numpy() # type: ignore
    d_min = np.min(data) # type: ignore
    d_max = np.max(data) # type: ignore


    # if the spectrum hasn't been base shifted to 0, then we do it to make plotting easier,
    # but we will lie about it on the y axis 
    if d_min < 0:
        data += np.abs(d_min)
        y_dmin = np.floor(d_min)
        y_dmax = np.ceil(d_max)
        ytx = np.abs(y_dmin) + y_dmax
        ticks = np.arange(0, ytx)
        labels = [str(x + y_dmin) for x in ticks]
        ax.set_yticks(ticks, labels=labels)
        ranking = np.repeat(ranking, len(labels), axis=0)

    ax.plot(data)  # type: ignore
    ax.set_ylabel("Wave Intensity")

    k = args.target.classification # type: ignore
    confidence = args.target.confidence #type: ignore
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

    if args.output == "show":
        plt.show()
    else:
        plt.savefig(args.output, dpi=300)


def surface_plot(
    args: CausalArgs, resp_maps: ResponsibilityMaps, target: Prediction, destination=None
):
    """plots a 3d surface plot"""
    img, _x, _y = plot_3d(args.path, resp_maps.get(target.classification), True)
    fig = plt.figure()

    # TODO enable visualisation of sub-responsibility maps
    keys = [target.classification]

    rows, cols = 1, len(keys)

    for i, k in enumerate(keys):
        ranking = resp_maps.get(k)
        if ranking is not None:
            ax = fig.add_subplot(rows, cols, i + 1, projection="3d")

            ax.plot_surface( #type: ignore
                _x, _y, np.atleast_2d(0), rstride=5, cstride=5, facecolors=img
            )  
            ax.plot_surface(_x, _y, resp_maps.get(k), alpha=0.4, cmap=mpl.colormaps[args.heatmap_colours])  # type: ignore
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
                    if "GB" in os.environ["LANG"]:
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

        if destination == "show":
            plt.show()
        else:
            plt.savefig(destination, bbox_inches="tight", dpi=300)


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


def save_image(explanation, data: Data, args: CausalArgs):
    if args.output is not None:
        name = args.output
    else:
        name = f"{args.target.classification}.png"  # type: ignore

    mask = None
    if data.mode == "RGB" or data.mode == "L":
        img = data.input.resize((data.model_height, data.model_width))

        if data.transposed:
            if data.mode == "RGB" or data.mode == "L":
                mask = (
                    explanation.squeeze()
                    .detach()
                    .cpu()
                    .numpy()
                    .transpose((1, 2, 0))
                )
        else:
            mask = explanation.squeeze().detach().cpu().numpy()

        if mask is not None:
            if args.raw:
                out = np.where(mask, img, args.mask_value).squeeze()
                out = Image.fromarray(out, data.mode)
                out.save(name)
                return out

            else:
                exp = np.where(mask, img, args.colour)
                exp = Image.fromarray(exp, data.mode)
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

                out.save(name)
                return out

