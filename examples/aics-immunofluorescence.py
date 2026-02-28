import marimo

__generated_with = "0.20.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import numpy.typing as npt
    import skimage.data
    from matplotlib import pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from skimage.util import img_as_float


    def get_data():
        data = skimage.data.cells3d()
        z = data.shape[0] // 2
        return img_as_float(data[z, ...])


    initial = get_data()


    def plot_two_images(im0: npt.NDArray, im1: npt.NDArray, *, title: str):
        fig, axes = plt.subplots(nrows=1, ncols=2)
        fig: Figure
        ax0: Axes = axes[0]
        ax0.imshow(im0)
        ax0.axis('off')

        ax1: Axes = axes[1]
        ax1.imshow(im1)
        ax1.axis('off')

        fig.suptitle(title)
        fig.tight_layout()
        return fig

    return Axes, Figure, initial, npt, plot_two_images, plt


@app.cell
def _(initial, plot_two_images):
    fig_initial = plot_two_images(initial[0], initial[1], title='Initial')
    fig_initial
    return (fig_initial,)


@app.cell
def _(mo):
    a01 = mo.ui.slider(start=0.0, stop=1.0, step=0.05, value=0.6, label="Mixing Coefficient a01")
    a10 = mo.ui.slider(start=0.0, stop=1.0, step=0.05, value=0.7, label="Mixing Coefficient a10")

    mo.vstack([a01, a10])
    return a01, a10


@app.cell
def _(initial, plot_two_images, a01, a10):
    import numpy as np

    # make up a mixing matrix
    mixing_matrix = np.array([[1.0, a01.value], [a10.value, 1.0]])
    mixed = np.einsum("ij,jkl->ikl", mixing_matrix, initial)

    fig_mixed = plot_two_images(mixed[0], mixed[1], title='Artificially mixed')
    return fig_mixed, mixed, mixing_matrix, np


@app.cell
def _(fig_mixed):
    fig_mixed
    return


@app.cell
def _(mixed):
    from picasso.unmixing import compute_unmixing_matrix
    from skimage.transform import downscale_local_mean

    mixed_downscaled = downscale_local_mean(mixed, (1, 2, 2))
    mat_iters = compute_unmixing_matrix(
        mixed_downscaled, verbose=True, return_iters=True
    )
    unmixing_matrix = mat_iters[-1]
    print(unmixing_matrix)
    return compute_unmixing_matrix, downscale_local_mean, mat_iters, mixed_downscaled, unmixing_matrix


@app.cell
def _(mixed, np, plot_two_images, unmixing_matrix):
    from picasso.unmixing import apply_unmixing_matrix

    unmixed = apply_unmixing_matrix(mixed, unmixing_matrix)
    np.clip(unmixed, 0, 1, out=unmixed)

    fig_unmixed = plot_two_images(unmixed[0], unmixed[1], title='Unmixed')
    return apply_unmixing_matrix, fig_unmixed, unmixed


@app.cell
def _(fig_unmixed):
    fig_unmixed
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Oddly, this implementation does not completely get rid of overlap...
    """)
    return


@app.cell
def _(Axes, Figure, mat_iters, plt):
    fig, ax = plt.subplots()
    fig: Figure
    ax: Axes
    ax.plot(mat_iters[:, 0, 0], label='a00')
    ax.plot(mat_iters[:, 0, 1], label='a01')
    ax.plot(mat_iters[:, 1, 0], label='a10')
    ax.plot(mat_iters[:, 1, 1], label='a11')
    fig.legend()
    fig.suptitle("Convergence of coefs")
    fig.tight_layout()
    fig
    return ax, fig


if __name__ == "__main__":
    app.run()
