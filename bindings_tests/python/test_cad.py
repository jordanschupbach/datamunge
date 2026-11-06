"""Owned CAD solids and code-driven scenes use the canonical native API."""
import gc
import json
import pytest
from pydatamunge import datamunge as dm


def test_owned_solid_scene_roundtrip(tmp_path):
    solid = dm.CadSolid.box(2, 3, 4)
    assert solid.valid()
    assert solid.volume() == pytest.approx(24)
    translated = solid.translated(.5, .5, .5)
    assert solid.intersected(translated).volume() == pytest.approx(13.125, abs=1e-6)
    view = dm.CadScene()
    identifier = view.add(solid, '#4080d9', 'block')
    del solid, translated
    gc.collect()
    assert view.object_count() == 1
    assert view.triangle_count() == 12
    view.size(96, 72)
    view.samples(2)
    view.render(str(tmp_path / 'block.png'))
    assert (tmp_path / 'block.png').read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    assert view.pick_object(48, 36) == identifier
    view.save_gltf(str(tmp_path / 'block.gltf'))
    assert json.loads((tmp_path / 'block.gltf').read_text())['asset']['version'] == '2.0'
    restored = dm.CadScene.from_file(str(tmp_path / 'block.gltf'))
    assert restored.object_count() == 1
    assert restored.triangle_count() == 12


def test_cad_argument_errors():
    with pytest.raises(ValueError, match='positive'):
        dm.CadSolid.box(0, 1, 1)
    view = dm.CadScene()
    with pytest.raises(ValueError):
        view.samples(5)
    with pytest.raises(ValueError):
        view.set_joint('unknown', 1)
    with pytest.raises(ValueError):
        view.add(dm.CadSolid.sphere(1), 'broken color')


def test_urdf_uses_native_kinematics(tmp_path):
    path = tmp_path / 'robot.urdf'
    path.write_text('''<robot name="fixture"><link name="base"/><link name="tip">
<visual><geometry><box size="1 1 1"/></geometry></visual></link>
<joint name="turn" type="revolute"><parent link="base"/><child link="tip"/>
<axis xyz="0 0 1"/><limit lower="-2" upper="2"/></joint></robot>''')
    view = dm.CadScene.from_urdf(str(path))
    assert list(view.joint_names()) == ['turn']
    view.set_joint('turn', .5)
    assert view.object_count() == 1
    with pytest.raises(ValueError, match='limits'):
        view.set_joint('turn', 3)
