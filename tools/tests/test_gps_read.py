"""read_gps: GPS from EXIF, and a drone's height above take-off from XMP in both the attribute form DJI writes and
the element form SplatWork's photo import writes."""
import numpy as np
from PIL import Image

from splatprep.gps import read_gps


def _jpeg(path, xmp=None):
    im = Image.fromarray(np.zeros((8, 8, 3), np.uint8))
    ex = im.getexif()
    ex.get_ifd(0x8825).update({1: "N", 2: (51, 30, 0.0), 3: "W", 4: (0, 6, 0.0), 5: 0, 6: 120.0})
    kw = {"exif": ex}
    if xmp:
        kw["xmp"] = xmp.encode()
    im.save(path, **kw)


def _xmp(body):
    return ('<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
            f'<rdf:Description xmlns:drone-dji="http://www.dji.com/drone-dji/1.0/" {body}</rdf:RDF></x:xmpmeta>')


def test_gps_altitude_from_exif(tmp_path):
    _jpeg(tmp_path / "a.jpg")
    lat, lon, alt = read_gps(tmp_path / "a.jpg")
    assert abs(lat - 51.5) < 1e-6 and abs(lon + 0.1) < 1e-6 and alt == 120.0


def test_relative_altitude_attribute_and_element_forms(tmp_path):
    _jpeg(tmp_path / "attr.jpg", _xmp('drone-dji:RelativeAltitude="+45.20"/>'))
    _jpeg(tmp_path / "elem.jpg", _xmp('><drone-dji:RelativeAltitude>+45.20</drone-dji:RelativeAltitude></rdf:Description>'))
    assert read_gps(tmp_path / "attr.jpg")[2] == 45.2
    assert read_gps(tmp_path / "elem.jpg")[2] == 45.2
