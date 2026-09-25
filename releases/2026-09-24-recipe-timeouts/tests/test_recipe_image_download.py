from unittest.mock import Mock, patch
import pytest
from test_saved_recipes_image_mirroring import load_app_module


def response(headers=None, chunks=None):
    value = Mock(headers={'Content-Type': 'image/png', **(headers or {})}, url='https://fixture.invalid/image')
    value.iter_content.return_value = iter(chunks or [b'fixture-image'])
    return value


def test_download_preserves_bytes_and_closes_stream():
    app = load_app_module(); remote = response(chunks=[b'first', b'second'])
    with patch.object(app.requests, 'get', return_value=remote) as get:
        result = app._load_remote_image(remote.url)
    assert result[0] == b'firstsecond' and get.call_args.kwargs['stream'] is True
    remote.close.assert_called_once()


def test_advertised_oversize_stops_before_reading_body():
    app = load_app_module(); remote = response({'Content-Length': str(26 * 1024 * 1024)})
    with patch.object(app.requests, 'get', return_value=remote):
        with pytest.raises(app.ServiceError) as error: app._load_remote_image(remote.url)
    assert error.value.status_code == 413
    remote.iter_content.assert_not_called(); remote.close.assert_called_once()


def test_unknown_length_is_still_bounded():
    app = load_app_module(); remote = response(chunks=[b'a' * (13 * 1024 * 1024), b'b' * (13 * 1024 * 1024)])
    with patch.object(app.requests, 'get', return_value=remote):
        with pytest.raises(app.ServiceError) as error: app._load_remote_image(remote.url)
    assert error.value.status_code == 413; remote.close.assert_called_once()


def test_slow_body_has_a_total_download_deadline():
    app = load_app_module(); remote = response()
    with patch.object(app.requests, 'get', return_value=remote), patch.object(app.time, 'monotonic', side_effect=[100, 121]):
        with pytest.raises(app.ServiceError) as error: app._load_remote_image(remote.url)
    assert error.value.status_code == 503; remote.close.assert_called_once()


@pytest.mark.parametrize('format, mime', [('JPEG', 'image/jpeg'), ('PNG', 'image/png'), ('WEBP', 'image/webp')])
@pytest.mark.parametrize('header', ['application/octet-stream', 'binary/octet-stream', ''])
def test_generic_binary_mime_requires_decodable_image(format, mime, header):
    from PIL import Image
    from io import BytesIO
    image = BytesIO(); Image.new('RGB', (3, 3), 'red').save(image, format=format)
    app = load_app_module(); remote = response({'Content-Type': header}, [image.getvalue()])
    with patch.object(app.requests, 'get', return_value=remote):
        data, content_type, _ = app._load_remote_image('https://fixture.invalid/file')
    assert data == image.getvalue() and content_type == mime
    remote.close.assert_called_once()


@pytest.mark.parametrize('data', [b'<html>Forbidden</html>', b'not an image', b'\xff\xd8\xfftruncated'])
def test_binary_mime_never_trusts_extension_or_corrupt_header(data):
    app = load_app_module(); remote = response({'Content-Type': 'application/octet-stream'}, [data])
    with patch.object(app.requests, 'get', return_value=remote):
        with pytest.raises(app.ServiceError) as error:
            app._load_remote_image('https://fixture.invalid/photo.jpg')
    assert error.value.status_code == 422
    remote.close.assert_called_once()


def test_mirror_uses_verified_mime_for_owned_original_and_thumbnail():
    from PIL import Image
    from io import BytesIO
    source = BytesIO(); Image.new('RGB', (3, 3)).save(source, format='JPEG')
    app = load_app_module(); remote = response({'Content-Type': 'application/octet-stream'}, [source.getvalue()])
    s3 = Mock()
    with patch.dict(app.os.environ, {'BUCKET_NAME': 'uploads-bucket'}), patch.object(app.requests, 'get', return_value=remote), patch.object(app.boto3, 'client', return_value=s3):
        result = app._mirror_recipe_image('fixture-owner', 'fixture-recipe', remote.url)
    assert result['image_url'].endswith('.jpg')
    assert s3.put_object.call_count == 2
    assert all(call.kwargs['ContentType'] == 'image/jpeg' for call in s3.put_object.call_args_list)
