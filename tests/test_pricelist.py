import pytest
from Munin.PriceList import create_pricelist, Pricelist, TimberPricelist
from Munin.PriceList.Data.Mellanskog_2013 import Mellanskog_2013_price_data

@pytest.fixture(scope="module")
def pricelist() -> Pricelist:
    return create_pricelist(Mellanskog_2013_price_data)

def test_pulpwood_prices(pricelist):
    # Check pulp prices for correctness
    assert pricelist.Pulp.getPulpwoodPrice("Pine") == 250
    assert pricelist.Pulp.getPulpwoodPrice("Spruce") == 265
    assert pricelist.Pulp.getPulpwoodPrice("Birch") == 250
    assert pricelist.Pulp.getPulpwoodPrice("NonExistingSpecies") == 200  # default price

def test_timber_prices_pine(pricelist):
    # Check timber prices for Pine species, diameter 20, Butt log
    pine_pricelist = pricelist.Timber["Pine"]
    butt_price = pine_pricelist[20].price_for_log_part(pine_pricelist.LogParts.Butt)
    middle_price = pine_pricelist[20].price_for_log_part(pine_pricelist.LogParts.Middle)
    top_price = pine_pricelist[20].price_for_log_part(pine_pricelist.LogParts.Top)

    assert butt_price == 575
    assert middle_price == 460
    assert top_price == 340

def test_timber_prices_spruce(pricelist):
    # Check timber prices for Spruce species, diameter 26
    spruce_pricelist = pricelist.Timber["Spruce"]
    butt_price = spruce_pricelist[26].price_for_log_part(spruce_pricelist.LogParts.Butt)
    middle_price = spruce_pricelist[26].price_for_log_part(spruce_pricelist.LogParts.Middle)
    top_price = spruce_pricelist[26].price_for_log_part(spruce_pricelist.LogParts.Top)

    assert butt_price == 575
    assert middle_price == 400
    assert top_price == 400

def test_common_attributes(pricelist):
    # Test some common attributes
    assert pricelist.TopDiameter == 5
    assert pricelist.HighStumpHeight == 4
    assert pricelist.PulpLogDiameter.Min == 5
    assert pricelist.PulpLogDiameter.Max == 60
    assert pricelist.TimberLogLength.Min == 34
    assert pricelist.TimberLogLength.Max == 55

def test_nonexistent_diameter(pricelist):
    # Prices for non-defined diameter should default to zero
    pine_pricelist = pricelist.Timber["Pine"]
    default_prices = pine_pricelist[99]

    assert default_prices.butt_price == 0
    assert default_prices.middle_price == 0
    assert default_prices.top_price == 0

def test_price_for_log_part_method(pricelist):
    spruce_pricelist = pricelist.Timber["Spruce"]

    # exact match
    assert spruce_pricelist.price_for_log_part(TimberPricelist.LogParts.Butt, 26) == 575
    
    # between classes (floors to nearest lower class, 26)
    assert spruce_pricelist.price_for_log_part(TimberPricelist.LogParts.Butt, 27.9) == 575
    
    # smaller than available classes (returns zero)
    assert spruce_pricelist.price_for_log_part(TimberPricelist.LogParts.Butt, 10) == 0

    # larger than available classes (returns highest available diameter price)
    assert spruce_pricelist.price_for_log_part(TimberPricelist.LogParts.Butt, 40) == 525
