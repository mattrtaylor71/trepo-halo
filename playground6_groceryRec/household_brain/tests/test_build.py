import json
import zipfile
from pathlib import Path
import pytest
from household_brain import build

def test_deterministic_package_and_overlay_preserves_other_work(tmp_path,monkeypatch):
    out=tmp_path/'service.zip'
    first=build.build(out);second=build.build(tmp_path/'second.zip')
    assert first['sha256']==second['sha256']
    with zipfile.ZipFile(out) as z:
        assert 'household_brain/runtime.py' in z.namelist()
        assert 'pymysql/__init__.py' in z.namelist()
        assert not any('test_' in n or 'operator.py' in n for n in z.namelist())
    original={name:('baseline '+name).encode() for name in ['store.py','api.py']}
    root=tmp_path/'repo';folder=root/'playground6_groceryRec/family_brain';folder.mkdir(parents=True)
    manifest=root/'docs/household-brain/imported-v1-manifest.json';manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({'playground6_groceryRec/family_brain/'+name:build.sha(data) for name,data in original.items()}))
    for name in original:(folder/name).write_text('candidate '+name)
    monkeypatch.setattr(build,'ROOT',root)
    base=tmp_path/'base.zip'
    with zipfile.ZipFile(base,'w') as z:
        for name,data in original.items():z.writestr('family_brain/'+name,data)
        z.writestr('app.py',b'current iOS-agent fix must survive')
    overlay=tmp_path/'candidate.zip'
    build.build(overlay,base,build.sha(base.read_bytes()),'python')
    with zipfile.ZipFile(overlay) as z:assert z.read('app.py')==b'current iOS-agent fix must survive'
    with pytest.raises(ValueError,match='base_package'):build.build(overlay,base,'incorrect','python')
    with zipfile.ZipFile(base,'w') as z:
        z.writestr('family_brain/api.py',b'new upstream fix');z.writestr('family_brain/store.py',original['store.py'])
    with pytest.raises(ValueError,match='target_source_drift'):build.build(overlay,base,build.sha(base.read_bytes()),'python')
