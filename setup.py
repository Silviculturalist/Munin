from setuptools import setup, find_packages

setup(
    name="Munin",
    version="0.2",
    author="Carl Vigren",
    author_email="carl.vigren@slu.se",
    url = "https://github.com/Silviculturalist/Munin",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "Munin.Geo.Humidity": ["humidity.shp"],
        "Munin.Geo.Climate": ["Klimat.shp"],
        "Munin.Geo.Coastline": ["SwedishCoastLine_NE_medium_clipped.shp"],
        "Munin.Geo.Counties" : ["RT_Dlanskod.shp"] 
    },
    python_requires = ">3.10",
    install_requires = [
        "scipy>=1.5.0",
        "geopandas",
        "numpy",
        "shapely",
        "pyproj"
    ]
)
