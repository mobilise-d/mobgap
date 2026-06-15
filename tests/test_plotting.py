import matplotlib.pyplot as plt

from mobgap.plotting import move_legend_outside


def test_move_legend_outside_allows_ncol_override():
    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot([0, 1], [0, 1], label="first")
    ax.plot([0, 1], [1, 0], label="second")
    ax.legend()

    move_legend_outside(fig, ax, ncol=1)

    legend = fig.legends[0]
    assert legend._ncols == 1
    plt.close(fig)
