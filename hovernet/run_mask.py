from mask_utils import make_auto_mask
import typer

import numpy as np


app = typer.Typer()

@app.command()
def main(slide : str = typer.Argument(..., help="Path to the high-resolution slide."),
         save : str = typer.Argument(..., help="Path to save the mask."),
         mask_level : int = typer.Option(
             3, help="Mask level. The higher the level, the faster the mask creation, the lower the quality."
             )
         ) -> np.ndarray:

    return make_auto_mask(slide, mask_level=mask_level, save=save)


if __name__ == "__main__":
    app()