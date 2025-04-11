"""Display registration."""

import itk
import matplotlib.pyplot as plt


def display_registration(fixed, moving, registered):
    """Three-panel comparison plot."""
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    ax[0].imshow(itk.array_from_image(fixed), cmap="gray")
    ax[0].set_title("Reference Frame")

    ax[1].imshow(itk.array_from_image(moving), cmap="gray")
    ax[1].set_title("Original Moving Frame")

    ax[2].imshow(itk.array_from_image(registered), cmap="gray")
    ax[2].set_title("Registered Result")

    plt.tight_layout()
    return fig
