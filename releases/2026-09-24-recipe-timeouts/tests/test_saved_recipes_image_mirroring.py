import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = Path(os.environ["RECIPE_CANDIDATE_SOURCE"]) / "app.py"
APP_DIR = str(APP_PATH.parent)


def load_app_module():
    module_name = f"saved_recipes_app_test_{uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, APP_PATH)
    module = importlib.util.module_from_spec(spec)
    env = {
        "BUCKET_NAME": "uploads-bucket",
    }
    with patch.dict(os.environ, env, clear=False):
        sys.path.insert(0, APP_DIR)
        try:
            spec.loader.exec_module(module)
        finally:
            sys.path.pop(0)
    return module


class SavedRecipesImageMirroringTests(unittest.TestCase):
    def setUp(self):
        self.app = load_app_module()

    def test_prepare_saved_recipe_image_fields_mirrors_to_owned_s3_url(self):
        response = Mock()
        response.headers = {"Content-Type": "image/jpeg"}
        response.content = b"jpeg-bytes"
        response.iter_content.return_value = iter([response.content])
        response.url = "https://cdn.example.com/recipe.jpg"
        response.raise_for_status = Mock()
        response.close = Mock()
        s3 = Mock()

        with patch.dict(os.environ, {"BUCKET_NAME": "uploads-bucket"}, clear=False), patch.object(
            self.app.requests, "get", return_value=response
        ), patch.object(self.app.boto3, "client", return_value=s3):
            image_fields = self.app._prepare_saved_recipe_image_fields(
                "owner-123",
                "recipe-123",
                {
                    "image_url": "https://cdn.example.com/recipe.jpg",
                    "image_urls": ["https://cdn.example.com/recipe.jpg"],
                },
                request_id="req-1",
            )

        self.assertEqual(
            image_fields["image_url"],
            "https://uploads-bucket.s3.amazonaws.com/recipe-images/owner-123/saved-recipes/recipe-123-1ec070879e7efd2a.jpg",
        )
        self.assertEqual(image_fields["image_urls"], [image_fields["image_url"]])
        self.assertEqual(image_fields["source_image_url"], "https://cdn.example.com/recipe.jpg")
        self.assertEqual(
            image_fields["image_storage_key"],
            "recipe-images/owner-123/saved-recipes/recipe-123-1ec070879e7efd2a.jpg",
        )
        self.assertEqual(s3.put_object.call_count, 2)
        original = s3.put_object.call_args_list[0].kwargs
        self.assertEqual(original['Body'], response.content)
        self.assertIn('recipe-123-source-', original['Key'])
        self.assertEqual(image_fields['source_image_urls'],
                         ['https://uploads-bucket.s3.amazonaws.com/' + original['Key']])

    def test_mirror_preserves_full_original_before_cropping_card(self):
        from PIL import Image
        from io import BytesIO
        image = BytesIO()
        Image.new('RGB', (1600, 800), '#123456').save(image, format='JPEG')
        response = Mock(headers={'Content-Type': 'image/jpeg'}, content=image.getvalue(),
                        url='https://fixture.invalid/wide.jpg')
        response.iter_content.return_value = iter([response.content])
        s3 = Mock()
        with patch.dict(os.environ, {'BUCKET_NAME': 'uploads-bucket'}), \
             patch.object(self.app.requests, 'get', return_value=response), \
             patch.object(self.app.boto3, 'client', return_value=s3):
            fields = self.app._mirror_recipe_image('owner', 'recipe', response.url)
        original, card = [call.kwargs for call in s3.put_object.call_args_list]
        self.assertEqual(original['Body'], image.getvalue())
        self.assertEqual(Image.open(BytesIO(original['Body'])).size, (1600, 800))
        width, height = Image.open(BytesIO(card['Body'])).size
        self.assertLessEqual(width / height, 1.2)
        self.assertNotEqual(fields['source_image_urls'][0], fields['image_url'])

    def test_failed_original_page_upload_is_not_silently_successful(self):
        with patch.object(self.app, '_upload_saved_recipe_source_image',
                          side_effect=['https://fixture.invalid/first.jpg', RuntimeError('fixture outage')]):
            with self.assertRaises(self.app.ServiceError) as error:
                self.app._upload_saved_recipe_source_images('owner', 'recipe', [
                    {'image_bytes': b'one', 'content_type': 'image/jpeg'},
                    {'image_bytes': b'two', 'content_type': 'image/jpeg'}])
        self.assertEqual(error.exception.status_code, 503)

    def test_ensure_owned_saved_recipe_image_refreshes_legacy_row(self):
        legacy_row = {
            "_id": "recipe-legacy",
            "image_url": "https://expired.example.com/old.jpg",
            "source_image_url": None,
            "source_url": "https://www.tiktok.com/@creator/video/123",
            "resolved_url": "https://www.tiktok.com/@creator/video/123",
        }
        mirrored = {
            "image_url": "https://uploads-bucket.s3.amazonaws.com/recipe-images/owner-123/saved-recipes/recipe-legacy-abcd1234.jpg",
            "image_urls": [
                "https://uploads-bucket.s3.amazonaws.com/recipe-images/owner-123/saved-recipes/recipe-legacy-abcd1234.jpg"
            ],
            "source_image_url": "https://fresh.example.com/new.jpg",
            "image_storage_key": "recipe-images/owner-123/saved-recipes/recipe-legacy-abcd1234.jpg",
        }

        with patch.dict(os.environ, {"BUCKET_NAME": "uploads-bucket"}, clear=False), patch.object(
            self.app, "_mirror_recipe_image", side_effect=[RuntimeError("expired"), mirrored]
        ), patch.object(
            self.app,
            "_extract_content",
            return_value={
                "image_url": "https://fresh.example.com/new.jpg",
                "image_urls": ["https://fresh.example.com/new.jpg"],
            },
        ), patch.object(self.app, "_update_saved_recipe_image_fields") as update_fields:
            refreshed = self.app._ensure_owned_saved_recipe_image(Mock(), "owner-123", legacy_row, request_id="req-2")

        self.assertEqual(refreshed["image_url"], mirrored["image_url"])
        self.assertEqual(refreshed["source_image_url"], mirrored["source_image_url"])
        self.assertEqual(refreshed["image_storage_key"], mirrored["image_storage_key"])
        update_fields.assert_called_once()


if __name__ == "__main__":
    unittest.main()
