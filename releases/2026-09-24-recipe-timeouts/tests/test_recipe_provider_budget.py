import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import Mock, patch

import pytest
from openai import OpenAI
from test_saved_recipes_image_mirroring import load_app_module


def test_actual_sdk_429_uses_only_three_calls_and_remains_retryable():
    app = load_app_module()
    calls = []
    class Server(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get('Content-Length', 0)))
            calls.append(self.path)
            body = json.dumps({'error': {'message': 'fixture quota', 'type': 'rate_limit_error'}}).encode()
            self.send_response(429)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *args): pass
    server = HTTPServer(('127.0.0.1', 0), Server)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = OpenAI(api_key='fixture', base_url='http://127.0.0.1:%d/v1' % server.server_port)
    token = app.recipe_batch_checkpoint._deadline.set(time.monotonic() + 95)
    try:
        with patch.object(app, 'OpenAI', return_value=client), patch.dict(app.os.environ, {'OPENAI_API_KEY': 'fixture'}), patch.object(app.time, 'sleep'):
            with pytest.raises(app.ServiceError) as exc:
                app._extract_recipe_fragment_from_image({'image_bytes': b'fixture', 'content_type': 'image/png'}, image_index=0)
        assert exc.value.status_code == 503 and len(calls) == 3
    finally:
        app.recipe_batch_checkpoint._deadline.reset(token)
        client.close()
        server.shutdown()
        server.server_close()


def test_completed_not_recipe_votes_remain_content_rejection():
    app = load_app_module()
    response = Mock()
    response.choices = [Mock(message=Mock(content='{"not_recipe":true,"transcribed_text":""}'))]
    client = Mock()
    client.chat.completions.create.return_value = response
    with patch.object(app, '_openai_client', return_value=client), patch.object(app.time, 'sleep'):
        with pytest.raises(app.ServiceError) as exc:
            app._extract_recipe_fragment_from_image({'image_bytes': b'fixture', 'content_type': 'image/png'})
    assert exc.value.status_code == 422


def test_provider_json_failure_is_retryable_not_unsuitable_content():
    app = load_app_module()
    client = Mock()
    client.chat.completions.create.side_effect = TimeoutError('fixture')
    with patch.object(app, '_openai_client', return_value=client), patch.object(app, '_report_backend_error'):
        with pytest.raises(app.ServiceError) as exc:
            app._refine_recipe_structured('Boil 1 cup of water and add couscous.')
    assert exc.value.status_code == 503


def test_grouping_outage_does_not_split_continuation_pages_in_durable_worker():
    app = load_app_module()
    client = Mock()
    client.chat.completions.create.side_effect = TimeoutError('fixture')
    token = app.recipe_batch_checkpoint._deadline.set(time.monotonic() + 95)
    try:
        with patch.object(app, '_openai_client', return_value=client):
            with pytest.raises(app.ServiceError) as exc:
                app._group_image_fragments([{'content': 'ingredients'}, {'content': 'steps'}])
        assert exc.value.status_code == 503
    finally:
        app.recipe_batch_checkpoint._deadline.reset(token)


def test_instagram_login_wall_moves_to_fallback_without_identical_retries():
    app = load_app_module()
    ydl = Mock()
    ydl.extract_info.side_effect = RuntimeError('Requested content is not available, rate-limit reached or login required. Use --cookies')
    manager = Mock(); manager.__enter__ = Mock(return_value=ydl); manager.__exit__ = Mock(return_value=False)
    with patch.object(app.yt_dlp, 'YoutubeDL', return_value=manager), patch.object(app, '_resolve_ytdlp_cookiefile', return_value=None), patch.object(app.time, 'sleep') as sleep:
        with pytest.raises(app.ServiceError):
            app._extract_ytdlp_info('https://www.instagram.com/reel/fixture/')
    assert ydl.extract_info.call_count == 1, 'Repeating the same unauthenticated request cannot supply missing login credentials'
    sleep.assert_not_called()


def test_login_wall_still_preserves_caption_and_media_from_next_provider():
    app = load_app_module()
    ydl = Mock(); ydl.extract_info.side_effect = RuntimeError('Requested content is not available, rate-limit reached or login required')
    manager = Mock(); manager.__enter__ = Mock(return_value=ydl); manager.__exit__ = Mock(return_value=False)
    fallback = Mock(return_value={'content':'1 cup oats. Simmer with milk.', 'title':'Porridge',
                                 'source':'apify', 'media_url':'https://media.example.test/fixture.mp4', 'warnings':[]})
    with patch.object(app.yt_dlp,'YoutubeDL',return_value=manager), patch.object(app,'_resolve_ytdlp_cookiefile',return_value=None), patch.object(app.time,'sleep') as sleep, patch.object(app,'_log_event'):
        result=app._run_provider_pipeline('https://www.instagram.com/reel/fixture/','https://www.instagram.com/reel/fixture/','instagram',
                                          [('yt-dlp',app._extract_ytdlp),('apify',fallback)],request_id='fixture')
    assert ydl.extract_info.call_count==1
    sleep.assert_not_called();fallback.assert_called_once()
    assert result['content']=='1 cup oats. Simmer with milk.'
    assert result['media_url']=='https://media.example.test/fixture.mp4' and result['source']=='apify'
