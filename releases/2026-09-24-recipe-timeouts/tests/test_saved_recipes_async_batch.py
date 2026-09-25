import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "saved_recipes_api" / "app.py"
APP_DIR = str(APP_PATH.parent)


def load_app_module():
    module_name = f"saved_recipes_app_async_batch_test_{uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, APP_PATH)
    module = importlib.util.module_from_spec(spec)
    env = {
        "BUCKET_NAME": "uploads-bucket",
        "OPENAI_API_KEY": "test-key",
        "JOBS_TABLE": "jobs-table",
    }
    with patch.dict(os.environ, env, clear=False):
        sys.path.insert(0, APP_DIR)
        try:
            spec.loader.exec_module(module)
        finally:
            sys.path.pop(0)
    return module


class SavedRecipesAsyncBatchTests(unittest.TestCase):
    def setUp(self):
        self.app = load_app_module()

    def test_post_saved_recipe_images_returns_accepted_job(self):
        fake_job = {
            "job_id": "job-123",
            "type": "saved_recipe_batch",
            "owner": "owner-123",
            "status": "PENDING",
            "image_count": 2,
            "result_count": 0,
            "created_at": "2026-04-13T21:00:00Z",
            "updated_at": "2026-04-13T21:00:00Z",
        }
        event = {
            "requestContext": {
                "domainName": "example.execute-api.us-east-1.amazonaws.com",
                "stage": "$default",
            }
        }

        with patch.object(self.app, "_enqueue_saved_recipe_batch_job", return_value=fake_job):
            response = self.app._post_saved_recipe(
                "owner-123",
                {
                    "images": [
                        {"image_url": "https://example.com/one.jpg"},
                        {"image_url": "https://example.com/two.jpg"},
                    ],
                    "_event": event,
                },
                request_id="req-123",
            )

        self.assertEqual(response["statusCode"], 202)
        body = json.loads(response["body"])
        self.assertEqual(body["job"]["job_id"], "job-123")
        self.assertEqual(body["job"]["status"], "pending")
        self.assertEqual(
            body["job"]["poll_url"],
            "https://example.execute-api.us-east-1.amazonaws.com/saved-recipes/owner-123/jobs/job-123",
        )

    def test_get_saved_recipe_batch_job_status_returns_completed_results(self):
        job = {
            "job_id": "job-123",
            "type": "saved_recipe_batch",
            "owner": "owner-123",
            "status": "COMPLETED",
            "image_count": 5,
            "result_count": 2,
            "created_at": "2026-04-13T21:00:00Z",
            "updated_at": "2026-04-13T21:01:00Z",
            "results": [
                {"recipe_id": "recipe-1", "deduped": False},
                {"recipe_id": "recipe-2", "deduped": True},
            ],
            "partial_errors": [{"image_indexes": [4], "error": "failed", "status_code": 422}],
        }
        conn = Mock()

        with patch.object(self.app, "_get_saved_recipe_batch_job", return_value=job), patch.object(
            self.app, "_mysql_conn", return_value=conn
        ), patch.object(self.app, "_ensure_saved_recipes_table"), patch.object(
            self.app,
            "_fetch_saved_recipe_batch_results",
            return_value=(
                [{"id": "recipe-1", "ingredients":["Milk"], "instructions":["Heat milk."]}, {"id": "recipe-2", "ingredients":["Bread"], "instructions":["Toast bread."]}],
                [{"recipe": {"id": "recipe-1", "ingredients":["Milk"], "instructions":["Heat milk."]}, "deduped": False}, {"recipe": {"id": "recipe-2", "ingredients":["Bread"], "instructions":["Toast bread."]}, "deduped": True}],
                [{"image_indexes": [4], "error": "failed", "status_code": 422}],
            ),
        ):
            response = self.app._get_saved_recipe_batch_job_status("owner-123", "job-123", request_id="req-1")

        self.assertEqual(response["statusCode"], 200)
        body = json.loads(response["body"])
        self.assertEqual(body["job"]["status"], "completed")
        self.assertEqual(body["count"], 2)
        self.assertEqual(body["results"][1]["deduped"], True)
        self.assertEqual(body["partial_errors"][0]["image_indexes"], [4])
        # _mysql_conn reuses the warm Lambda connection; reads/jobs leave it open.
        conn.close.assert_not_called()

    def test_handle_async_saved_recipe_batch_task_marks_job_complete(self):
        job = {
            "job_id": "job-123",
            "type": "saved_recipe_batch",
            "owner": "owner-123",
            "status": "PENDING",
            "payload_s3_key": "recipe-images/owner-123/saved-recipes/jobs/job-123-submission.json",
        }
        conn = Mock()

        with patch.object(self.app, "_get_saved_recipe_batch_job", return_value=job), patch.object(
            self.app, "_update_saved_recipe_batch_job", side_effect=[{"status": "RUNNING"}, {"status": "COMPLETED"}]
        ) as update_mock, patch.object(
            self.app,
            "_load_saved_recipe_batch_payload",
            return_value={"images": [{"image_url": "https://example.com/one.jpg"}]},
        ), patch.object(self.app, "_mysql_conn", return_value=conn), patch.object(
            self.app, "_ensure_saved_recipes_table"
        ), patch.object(
            self.app,
            "_process_saved_recipe_image_batch",
            return_value=([{"recipe_id": "recipe-1", "deduped": False, "image_indexes": [0]}], []),
        ):
            result = self.app._handle_async_saved_recipe_batch_task(
                {"owner": "owner-123", "job_id": "job-123"},
                request_id="req-async",
            )

        self.assertEqual(result["ok"], True)
        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(update_mock.call_count, 2)
        # _mysql_conn reuses the warm Lambda connection; reads/jobs leave it open.
        conn.close.assert_not_called()


if __name__ == "__main__":
    unittest.main()
