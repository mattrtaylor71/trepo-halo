import hashlib,importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('release_build',Path(__file__).with_name('build.py'))
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
def test_unknown_serving_source_change_stops_build():
    with pytest.raises(ValueError,match='unreviewed_source_drift'):b.reconcile('function','module',b'foreign',b'base',b'new')
def test_reviewed_variant_preserves_distinct_guard_and_applies_only_repair(monkeypatch):
    base=b'guard=worker\n\n\nquantity=old\n';variant=base.replace(b'worker',b'other-protected-entrypoint');repair=base.replace(b'old',b'fixed')
    monkeypatch.setattr(b,'REVIEWED_VARIANTS',{'function':{'module':hashlib.sha256(variant).hexdigest()}})
    assert b.reconcile('function','module',variant,base,repair)==variant.replace(b'old',b'fixed')
def test_reviewed_variant_overlap_fails_instead_of_overwriting(monkeypatch):
    variant=b'quantity=foreign\n';monkeypatch.setattr(b,'REVIEWED_VARIANTS',{'function':{'module':hashlib.sha256(variant).hexdigest()}})
    with pytest.raises(ValueError,match='variant_merge_conflict'):b.reconcile('function','module',variant,b'quantity=base\n',b'quantity=repair\n')

def test_smoke_extraction_keeps_decoder_executable_and_rejects_escape(tmp_path):
    import io,zipfile
    spec=importlib.util.spec_from_file_location('release_smoke',Path(__file__).with_name('smoke.py'))
    smoke=importlib.util.module_from_spec(spec);spec.loader.exec_module(smoke)
    data=io.BytesIO()
    with zipfile.ZipFile(data,'w') as z:
        info=zipfile.ZipInfo('bin/ffmpeg');info.external_attr=0o100755 << 16
        z.writestr(info,b'fixture executable')
    with zipfile.ZipFile(data) as z:smoke.extract_with_modes(z,tmp_path)
    assert (tmp_path/'bin/ffmpeg').stat().st_mode & 0o777 == 0o755
    data=io.BytesIO()
    with zipfile.ZipFile(data,'w') as z:z.writestr('../escape',b'no')
    with zipfile.ZipFile(data) as z:
        with pytest.raises(ValueError,match='unsafe_zip_path'):smoke.extract_with_modes(z,tmp_path)
