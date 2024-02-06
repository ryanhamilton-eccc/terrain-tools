import click
import ee
import tagee

from dataclasses import dataclass, field
from typing import Callable

# NOTE: DO NOT ATTEMPT TO TEST (I.E. ATTEMPT TO TAKE THIS OUT SIDE OF OF API) THIS FILE,
# IT IS NOT VERY TESTABLE, WE WILL HAVE TO ASSUME THAT THE
# FUNCTIONS WORK AS INTENDED


####################################################################################################
# Smoothing Algorithms for Terrain Analysis
####################################################################################################
def gaussian_filter(
    radius: float = 3,
    sigma: int = 2,
    units: str = None,
    normalize: bool = True,
    magnitude: float = 1.0,
) -> Callable:
    units = "pixels" if units is None else units
    return lambda image: image.convolve(
        ee.Kernel.gaussian(
            radius=radius,
            sigma=sigma,
            units=units,
            normalize=normalize,
            magnitude=magnitude,
        )
    )


# TODO: refactor and hard code the filter, we only use default values
def perona_malik(K=3.5, iterations=10, method=2) -> Callable:
    """translated from this example here https://mygeoblog.com/2021/01/22/perona-malik-filter/

    Args:
        k (float, optional): _description_. Defaults to 3.5.
        iter (int, optional): _description_. Defaults to 10.
        method (int, optional): _description_. Defaults to 2.

    Returns:
        callable: _description_
    """

    def wrapper(img: ee.Image):
        dxW = ee.Kernel.fixed(3, 3, [[0, 0, 0], [1, -1, 0], [0, 0, 0]])

        dxE = ee.Kernel.fixed(3, 3, [[0, 0, 0], [0, -1, 1], [0, 0, 0]])

        dyN = ee.Kernel.fixed(3, 3, [[0, 1, 0], [0, -1, 0], [0, 0, 0]])

        dyS = ee.Kernel.fixed(3, 3, [[0, 0, 0], [0, -1, 0], [0, 1, 0]])

        lamb = 0.2

        k1 = ee.Image(-1.0 / K)
        k2 = ee.Image(K).multiply(ee.Image(K))

        for _ in range(0, iterations):
            dI_W = img.convolve(dxW)
            dI_E = img.convolve(dxE)
            dI_N = img.convolve(dyN)
            dI_S = img.convolve(dyS)

            if method == 1:
                cW = dI_W.multiply(dI_W).multiply(k1).exp()
                cE = dI_E.multiply(dI_E).multiply(k1).exp()
                cN = dI_N.multiply(dI_N).multiply(k1).exp()
                cS = dI_S.multiply(dI_S).multiply(k1).exp()

                img = img.add(
                    ee.Image(lamb).multiply(
                        cN.multiply(dI_N)
                        .add(cS.multiply(dI_S))
                        .add(cE.multiply(dI_E))
                        .add(cW.multiply(dI_W))
                    )
                )

            else:
                cW = ee.Image(1.0).divide(
                    ee.Image(1.0).add(dI_W.multiply(dI_W).divide(k2))
                )
                cE = ee.Image(1.0).divide(
                    ee.Image(1.0).add(dI_E.multiply(dI_E).divide(k2))
                )
                cN = ee.Image(1.0).divide(
                    ee.Image(1.0).add(dI_N.multiply(dI_N).divide(k2))
                )
                cS = ee.Image(1.0).divide(
                    ee.Image(1.0).add(dI_S.multiply(dI_S).divide(k2))
                )

                img = img.add(
                    ee.Image(lamb).multiply(
                        cN.multiply(dI_N)
                        .add(cS.multiply(dI_S))
                        .add(cE.multiply(dI_E))
                        .add(cW.multiply(dI_W))
                    )
                )

        return img

    return wrapper


####################################################################################################
# Main Terrain Analysis Function and Data Classes
####################################################################################################
@dataclass
class Gaussian:
    dataset: ee.Image
    geom: ee.Geometry
    elevation_band: str = field(default="elevation")
    bands: list[str] = field(
        default_factory=lambda: ["Elevation", "Slope", "GaussianCurvature"]
    )
    filter_func: Callable = field(
        default=gaussian_filter()
    )  # TODO need to check paramaters


@dataclass
class PeronaMalik:
    dataset: ee.Image
    geom: ee.Geometry
    elevation_band: str = field(default="elevation")
    bands: list[str] = field(
        default_factory=lambda: [
            "HorizontalCurvature",
            "VerticalCurvature",
            "MeanCurvature",
        ]
    )
    filter_func: Callable = field(default=perona_malik())


def compute_terrain_products(data: Gaussian | PeronaMalik) -> ee.Image:
    """
    needs to take a terrain dataset and a region of interest and compute the following products:
    but we need to run the smoothing algorithms first on select bands. Meaning we will have to
    run the ta 2 times and combine the results into a single image at the end
    """
    # build a rectangle around the dataset
    geom = data.geom
    if isinstance(geom, ee.FeatureCollection):
        geom = geom.geometry()
    else:
        geom = geom

    coords = geom.bounds().coordinates()

    listCoords = ee.Array.cat(coords, 1)
    xCoords = listCoords.slice(1, 0, 1)
    yCoords = listCoords.slice(1, 1, 2)

    xMin = xCoords.reduce("min", [0]).get([0, 0])
    xMax = xCoords.reduce("max", [0]).get([0, 0])
    yMin = yCoords.reduce("min", [0]).get([0, 0])
    yMax = yCoords.reduce("max", [0]).get([0, 0])

    rectangle = ee.Geometry.Rectangle(xMin, yMin, xMax, yMax)

    dataset = data.filter_func(dataset)
    dataset = tagee.terrainAnalysis(
        dem=dataset.select(data.elevation_band), geom=rectangle
    ).select(data.bands)
    return dataset
