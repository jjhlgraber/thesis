import os
import torch
from matplotlib import image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle, Polygon
from itertools import product
import pickle


def generate_custom_card(
    number_index, color_index, pattern_index, shape_index, data_dir=None
):

    numbers = [1, 2, 3, 4]
    colors = ["red", "green", "purple", "yellow"]
    patterns = ["empty", "striped_vertical", "solid", "striped_horizontal"]
    shapes = ["diamond", "oval", "bar", "tie"]

    number = numbers[number_index]
    color = colors[color_index]
    pattern = patterns[pattern_index]
    shape = shapes[shape_index]

    plot_width = 4
    plot_heigth = 1.4 * plot_width

    # don't ask about this
    magic_correction_factor = 38.5
    magic_correction_factor = 38.5 / plot_width
    # magic_correction_factor = 3.85
    dpi = (50 * 70) / (plot_width * plot_heigth * magic_correction_factor)

    fig, ax = plt.subplots(
        figsize=(plot_width, plot_heigth), dpi=dpi
    )  # Adjusted figsize for 50x70 pixels
    ax.set_xlim([0, plot_width])
    ax.set_ylim([0, plot_heigth])
    ax.axis("off")

    colors_plt_codes = {
        "red": "r",
        "green": "g",
        "purple": "purple",
        "yellow": "yellow",
    }

    colors_plt_code = colors_plt_codes[color]

    y_spacing = plot_heigth / (number + 1) + 0.05 * plot_heigth

    biases = [
        plot_heigth * 0.05,
        plot_heigth * 0.075,
        plot_heigth * 0.1,
        plot_heigth * 0.125,
    ]
    for i in range(number):
        x = plot_width * 0.5

        y = (i + 1) * y_spacing - biases[number - 1]

        if shape == "diamond":
            shape_object = Polygon(
                [
                    [x - 0.5 * x, y],
                    [x, y + plot_heigth * 0.1],
                    [x + 0.5 * x, y],
                    [x, y - plot_heigth * 0.1],
                ]
            )
        elif shape == "oval":
            shape_object = Ellipse(
                (x, y), width=plot_width * 0.75, height=plot_heigth * 0.2
            )
        elif shape == "bar":
            shape_object = Rectangle(
                (x - 0.75 * x, y - plot_heigth * 0.1),
                width=plot_width * 0.75,
                height=plot_heigth * 0.2,
            )
        elif shape == "tie":
            shape_object = Polygon(
                [
                    [x - 0.5 * x, y - plot_heigth * 0.1],
                    [x - 0.5 * x, y + plot_heigth * 0.1],
                    [x + 0.5 * x, y - plot_heigth * 0.1],
                    [x + 0.5 * x, y + plot_heigth * 0.1],
                ]
            )

        # Set shading
        if pattern == "solid":
            shape_object.set_facecolor(colors_plt_code)
        elif pattern == "striped_vertical":
            shape_object.set_facecolor("none")
            shape_object.set_edgecolor(colors_plt_code)
            shape_object.set_hatch("||")
        elif pattern == "empty":
            shape_object.set_facecolor("none")
            shape_object.set_edgecolor(colors_plt_code)
        elif pattern == "striped_horizontal":
            shape_object.set_facecolor("none")
            shape_object.set_edgecolor(colors_plt_code)
            shape_object.set_hatch("--")

        ax.add_patch(shape_object)

    # Draw card border
    rect = plt.Rectangle(
        (0, 0.01),
        plot_width - 0.05,
        plot_heigth - 0.05,
        linewidth=2,
        edgecolor="black",
        facecolor="none",
    )
    ax.add_patch(rect)

    if data_dir:
        # Save plt fig as png
        file_path = os.path.join(
            data_dir,
            f"setcard_{number_index}{color_index}{pattern_index}{shape_index}.png",
        )
        plt.savefig(
            file_path,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.0,
        )
    else:
        plt.show()
    plt.close(fig)


def generate_all_custom_cards(data_dir="data/custom_cards"):
    all_cards = product(range(4), repeat=4)
    for card in all_cards:
        number_index, color_index, pattern_index, shape_index = card
        generate_custom_card(
            number_index, color_index, pattern_index, shape_index, data_dir
        )


def save_custom_cards_as_tensor(data_dir="data/custom_cards"):
    feature_states = [0, 1, 2, 3]
    n_cards = 256
    n_image_channels, height, width = 4, 70, 50
    card_feature_vectors = torch.LongTensor(list(product(feature_states, repeat=4)))

    cards_tensor = torch.empty(n_cards, n_image_channels, height, width)
    features_to_index = {}
    for i, card in enumerate(card_feature_vectors.tolist()):
        number_index, color_index, pattern_index, shape_index = card
        features_to_index[(number_index, color_index, pattern_index, shape_index)] = i
        image_file_path = os.path.join(
            data_dir,
            f"setcard_{number_index}{color_index}{pattern_index}{shape_index}.png",
        )
        im = mpimg.imread(image_file_path)
        # cards are vertical, but loaded horizontally
        cards_tensor[i] = torch.from_numpy(im).permute(2, 0, 1)

    save_tensor_file_path = os.path.join(
        data_dir,
        "custom_setcards.pt",
    )
    torch.save(cards_tensor, save_tensor_file_path)

    save_dict_file_path = os.path.join(
        data_dir,
        "features_to_index.pkl",
    )
    with open(save_dict_file_path, "wb") as f:
        pickle.dump(features_to_index, f)
