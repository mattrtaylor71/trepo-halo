import base64
import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "saved_recipes_api" / "app.py"
APP_DIR = str(APP_PATH.parent)


def load_app_module():
    module_name = f"saved_recipes_app_manual_test_{uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, APP_PATH)
    module = importlib.util.module_from_spec(spec)
    env = {
        "BUCKET_NAME": "uploads-bucket",
        "OPENAI_API_KEY": "test-key",
    }
    with patch.dict(os.environ, env, clear=False):
        sys.path.insert(0, APP_DIR)
        try:
            spec.loader.exec_module(module)
        finally:
            sys.path.pop(0)
    return module


class SavedRecipesManualInputsTests(unittest.TestCase):
    def setUp(self):
        self.app = load_app_module()

    def test_build_saved_recipe_input_accepts_content_alias(self):
        submission = self.app._build_saved_recipe_input({"text": "Grandma's chili"})

        self.assertEqual(submission["kind"], "content")
        self.assertEqual(submission["content"], "Grandma's chili")

    def test_build_saved_recipe_input_rejects_multiple_modes(self):
        with self.assertRaises(self.app.ServiceError) as ctx:
            self.app._build_saved_recipe_input(
                {"url": "https://example.com/recipe", "content": "duplicate input"}
            )

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("exactly one", str(ctx.exception))

    def test_build_saved_recipe_input_accepts_images_array(self):
        submission = self.app._build_saved_recipe_input(
            {
                "images": [
                    {"image_url": "https://example.com/recipe-1.jpg"},
                    {"image_base64": base64.b64encode(b"image-two").decode("ascii"), "image_content_type": "image/png"},
                ]
            }
        )

        self.assertEqual(submission["kind"], "images")
        self.assertEqual(len(submission["images"]), 2)

    def test_build_text_recipe_extraction_uses_stable_app_url(self):
        extraction = self.app._build_text_recipe_extraction("2 eggs\n1 tortilla\nCook and fold.")

        self.assertEqual(extraction["platform"], "text")
        self.assertTrue(extraction["resolved_url"].startswith("app://saved-recipes/text/"))
        self.assertEqual(extraction["content"], "2 eggs\n1 tortilla\nCook and fold.")

    def test_prepare_image_for_vision_converts_heic(self):
        converted = b"jpeg-bytes"

        with patch.object(
            self.app,
            "_normalize_image_submission",
            return_value={
                "kind": "bytes",
                "image_bytes": b"heic-bytes",
                "content_type": "image/heic",
                "fingerprint": "orig",
            },
        ), patch.object(
            self.app,
            "_convert_image_bytes_to_jpeg",
            return_value=(converted, "image/jpeg"),
        ) as convert_mock:
            prepared = self.app._prepare_image_for_vision({"image_base64": "unused"})

        self.assertEqual(prepared["content_type"], "image/jpeg")
        self.assertEqual(prepared["image_bytes"], converted)
        convert_mock.assert_called_once()

    def test_extract_recipe_from_image_supports_base64_payload(self):
        png_bytes = b"\x89PNG\r\n\x1a\nfake"
        image_base64 = base64.b64encode(png_bytes).decode("ascii")
        fake_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"transcribed_text":"Title: Pancakes\\nIngredients:\\n- flour","title_hint":"Pancakes","author_name":"","not_recipe":false}'
                    )
                )
            ]
        )
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=Mock(return_value=fake_response))
            )
        )

        with patch.object(self.app, "_openai_client", return_value=fake_client):
            extraction, image_input = self.app._extract_recipe_from_image(
                {"image_base64": image_base64, "image_content_type": "image/png"},
                request_id="req-image",
            )

        self.assertEqual(extraction["platform"], "image")
        self.assertEqual(extraction["title"], "Pancakes")
        self.assertTrue(extraction["resolved_url"].startswith("app://saved-recipes/image/"))
        self.assertEqual(image_input["kind"], "bytes")
        self.assertEqual(image_input["content_type"], "image/png")

    def test_extract_ytdlp_info_retries_explicit_transient_throttling(self):
        ydl = MagicMock()
        ydl.extract_info.side_effect = [
            RuntimeError("HTTP Error 429: Too Many Requests"),
            RuntimeError("HTTP Error 429: Too Many Requests"),
            {"description": "Recipe caption"},
        ]
        ydl_factory = MagicMock()
        ydl_factory.__enter__.return_value = ydl
        ydl_factory.__exit__.return_value = False

        with patch.object(self.app.yt_dlp, "YoutubeDL", return_value=ydl_factory), patch.object(
            self.app.time, "sleep"
        ) as sleep_mock:
            info = self.app._extract_ytdlp_info("https://www.instagram.com/reel/test/")

        self.assertEqual(info["description"], "Recipe caption")
        self.assertEqual(ydl.extract_info.call_count, 3)
        self.assertEqual(sleep_mock.call_count, 2)

    def test_extract_ytdlp_info_uses_cookiefile_from_base64_env(self):
        ydl = MagicMock()
        ydl.extract_info.return_value = {"description": "Recipe caption"}
        ydl_factory = MagicMock()
        ydl_factory.__enter__.return_value = ydl
        ydl_factory.__exit__.return_value = False
        cookie_bytes = b"# Netscape HTTP Cookie File\n.instagram.com\tTRUE\t/\tTRUE\t0\tsessionid\tabc\n"

        with patch.dict(os.environ, {"YTDLP_COOKIEFILE_B64": base64.b64encode(cookie_bytes).decode("ascii")}, clear=False), patch.object(
            self.app.yt_dlp, "YoutubeDL", return_value=ydl_factory
        ) as youtube_dl_mock:
            info = self.app._extract_ytdlp_info("https://www.instagram.com/reel/test/")

        self.assertEqual(info["description"], "Recipe caption")
        options = youtube_dl_mock.call_args.args[0]
        self.assertIn("cookiefile", options)
        self.assertTrue(Path(options["cookiefile"]).exists())
        self.assertEqual(Path(options["cookiefile"]).read_bytes(), cookie_bytes)

    def test_resolve_ytdlp_cookiefile_uses_existing_path(self):
        with patch.object(self.app.Path, "exists", return_value=True):
            with patch.dict(os.environ, {"YTDLP_COOKIEFILE": "/tmp/cookies.txt"}, clear=False):
                cookiefile = self.app._resolve_ytdlp_cookiefile()

        self.assertEqual(cookiefile, "/tmp/cookies.txt")

    def test_extract_instagram_apify_parses_caption_and_image(self):
        response = Mock()
        response.json.return_value = [
            {
                "url": "https://www.instagram.com/reel/test/",
                "caption": "Ingredients: pasta\nInstructions: boil",
                "displayUrl": "https://example.com/reel.jpg",
                "ownerUsername": "chef",
            }
        ]
        response.raise_for_status.return_value = None

        with patch.dict(os.environ, {"APIFY_TOKEN": "token"}, clear=False), patch.object(
            self.app.requests, "post", return_value=response
        ) as post_mock:
            extraction = self.app._extract_instagram_apify(
                "https://www.instagram.com/reel/test/",
                request_id="req-apify",
            )

        self.assertEqual(extraction["source"], "apify")
        self.assertEqual(extraction["caption_field"], "caption")
        self.assertEqual(extraction["image_url"], "https://example.com/reel.jpg")
        self.assertEqual(extraction["author_name"], "chef")
        post_mock.assert_called_once()

    def test_analyze_extraction_uses_apify_fallback_before_preview_ocr(self):
        extraction = {
            "platform": "instagram",
            "resolved_url": "https://www.instagram.com/reel/test/",
            "content": "Quick caption",
            "caption": "Quick caption",
            "source": "yt-dlp",
            "image_url": "",
            "image_urls": [],
            "warnings": [],
        }

        with patch.object(
            self.app,
            "_recipe_response_from_content",
            side_effect=[
                ({"recipe": "Not enough recipe information.", "model": "gpt-4.1-mini"}, {"not_enough": True}),
                ({"recipe": "Title:\nApify Pasta", "model": "gpt-4.1-mini"}, {"not_enough": False, "ingredients": ["pasta"], "instructions": ["Boil pasta."]}),
            ],
        ), patch.object(
            self.app,
            "_extract_instagram_apify",
            return_value={
                "content": "Ingredients: pasta\nInstructions: boil",
                "source": "apify",
                "image_url": "https://example.com/reel.jpg",
                "image_urls": ["https://example.com/reel.jpg"],
                "author_name": "chef",
                "caption_field": "caption",
            },
        ), patch.object(self.app, "_extract_recipe_from_social_preview_images") as preview_mock, patch.object(
            self.app, "_transcribe_audio_from_url"
        ) as audio_mock:
            result, structured = self.app._analyze_extraction(extraction, request_id="req-apify")

        self.assertEqual(result["recipe_source_used"], "content+apify")
        self.assertIn("Apify Instagram extraction", result["warnings"][0])
        self.assertFalse(structured["not_enough"])
        self.assertEqual(extraction["source"], "yt-dlp+apify")
        self.assertEqual(extraction["image_url"], "https://example.com/reel.jpg")
        preview_mock.assert_not_called()
        audio_mock.assert_not_called()

    def test_analyze_extraction_uses_preview_ocr_fallback_before_audio(self):
        extraction = {
            "platform": "instagram",
            "resolved_url": "https://www.instagram.com/reel/test/",
            "content": "Quick caption",
            "image_url": "https://example.com/preview.jpg",
            "image_urls": ["https://example.com/preview.jpg"],
            "warnings": [],
        }

        with patch.object(
            self.app,
            "_recipe_response_from_content",
            side_effect=[
                ({"recipe": "Not enough recipe information.", "model": "gpt-4.1-mini"}, {"not_enough": True}),
                ({"recipe": "Title:\nPreview Pasta", "model": "gpt-4.1-mini"}, {"not_enough": False, "ingredients": ["pasta"], "instructions": ["Boil pasta."]}),
            ],
        ), patch.object(
            self.app,
            "_extract_instagram_apify",
            side_effect=RuntimeError("APIFY_TOKEN is not set."),
        ), patch.object(
            self.app,
            "_extract_recipe_from_social_preview_images",
            return_value={"content": "Ingredients: pasta\nInstructions: boil", "title_hint": "Preview Pasta"},
        ), patch.object(self.app, "_transcribe_audio_from_url") as audio_mock:
            result, structured = self.app._analyze_extraction(extraction, request_id="req-preview")

        self.assertEqual(result["recipe_source_used"], "content+image_ocr")
        self.assertTrue(any("preview-image OCR" in warning for warning in result["warnings"]))
        self.assertFalse(structured["not_enough"])
        audio_mock.assert_not_called()

    def test_group_image_fragments_uses_model_clusters(self):
        fake_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"clusters":[{"image_indexes":[0,1],"title_hint":"Recipe A"},{"image_indexes":[2],"title_hint":"Recipe B"}],"discard_indexes":[]}'
                    )
                )
            ]
        )
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=Mock(return_value=fake_response))
            )
        )
        fragments = [
            {"image_index": 0, "title_hint": "A", "content": "A1"},
            {"image_index": 1, "title_hint": "A", "content": "A2"},
            {"image_index": 2, "title_hint": "B", "content": "B1"},
        ]

        with patch.object(self.app, "_openai_client", return_value=fake_client):
            clusters = self.app._group_image_fragments(fragments, request_id="req-group")

        self.assertEqual(len(clusters), 2)
        self.assertEqual([fragment["image_index"] for fragment in clusters[0]["fragments"]], [0, 1])
        self.assertEqual([fragment["image_index"] for fragment in clusters[1]["fragments"]], [2])

    def test_build_saved_recipe_post_response_for_multiple_recipes(self):
        response = self.app._build_saved_recipe_post_response(
            "owner-123",
            [
                {"recipe": {"id": "one", "title": "Recipe One", "ingredients":["Milk"], "instructions":["Heat milk."]}, "deduped": False},
                {"recipe": {"id": "two", "title": "Recipe Two", "ingredients":["Bread"], "instructions":["Toast bread."]}, "deduped": True},
            ],
            partial_errors=[{"image_index": 2, "error": "failed", "status_code": 422}],
        )

        self.assertEqual(response["statusCode"], 201)
        body = json.loads(response["body"])
        self.assertEqual(body["count"], 2)
        self.assertEqual(len(body["recipes"]), 2)
        self.assertEqual(body["results"][1]["deduped"], True)
        self.assertEqual(body["partial_errors"][0]["image_index"], 2)

    def test_prepare_generated_saved_recipe_image_fields_uploads_generated_image(self):
        s3 = Mock()
        recipe = {
            "title": "Tomato Pasta",
            "ingredients": ["pasta", "tomato"],
            "instructions": ["Boil pasta", "Add sauce"],
            "notes": [],
        }

        with patch.dict(os.environ, {"BUCKET_NAME": "uploads-bucket"}, clear=False), patch.object(
            self.app, "_generate_recipe_image", return_value=b"jpeg-bytes"
        ), patch.object(self.app.boto3, "client", return_value=s3):
            fields = self.app._prepare_generated_saved_recipe_image_fields(
                "owner-123",
                "recipe-123",
                recipe,
                source_image_urls=["https://uploads-bucket.s3.amazonaws.com/recipe-images/source.jpg"],
                request_id="req-gen",
            )

        self.assertTrue(fields["image_url"].startswith("https://uploads-bucket.s3.amazonaws.com/recipe-images/owner-123/saved-recipes/recipe-123-generated-"))
        self.assertEqual(fields["image_urls"], [fields["image_url"]])
        self.assertEqual(
            fields["source_image_url"],
            "https://uploads-bucket.s3.amazonaws.com/recipe-images/source.jpg",
        )
        s3.put_object.assert_called_once()

    def test_saved_recipe_fallback_cannot_upgrade_insufficient_quantity(self):
        context = {"kitchen_version": 1, "kitchen_candidates": [{
            "item_id": "fixture-beef", "display_name": "ground beef", "tokens": ["ground", "beef"],
            "quantity_value": 1, "quantity_unit": "oz",
        }]}
        recipe = {"ingredients": ["16 oz ground beef"]}
        result = self.app.recipe_inventory_llm.deterministic_availability(recipe, context)
        merged = self.app._local_deterministic_merge(result, context)
        self.assertFalse(merged["can_make_exact"])
        self.assertEqual(merged["missing_ingredients"], ["16 oz ground beef"])
        self.assertEqual(merged["insufficient_quantity_ingredients"], ["16 oz ground beef"])

    def test_compute_recipe_availability_matches_kitchen_items_and_pantry(self):
        kitchen_context = {
            "kitchen_version": 7,
            "confirmed_staples": ["salt", "unsalted butter"],
            "kitchen_candidates": [
                {
                    "item_id": "item-apple",
                    "display_name": "organic honeycrisp apples 6-pack",
                    "tokens": ["honeycrisp", "apple"],
                },
                {
                    "item_id": "item-turkey",
                    "display_name": "Jennie-O 93 lean ground turkey",
                    "tokens": ["ground", "turkey"],
                },
            ],
        }

        availability = self.app._compute_recipe_availability(
            ["apples", "ground turkey", "salt", "bison"],
            kitchen_context,
        )

        self.assertEqual(availability["kitchen_version"], 7)
        self.assertFalse(availability["can_make_exact"])
        self.assertEqual(availability["matched_count"], 3)
        self.assertEqual(availability["missing_count"], 1)
        self.assertEqual(availability["missing_ingredients"], ["bison"])
        self.assertEqual(
            [item["match_status"] for item in availability["ingredient_matches"]],
            ["have", "have", "pantry", "missing"],
        )
        self.assertEqual(
            availability["ingredient_matches"][0]["matched_kitchen_items"][0]["display_name"],
            "organic honeycrisp apples 6-pack",
        )

    def test_compute_recipe_availability_rejects_conflicting_subset_match_and_counts_staples(self):
        availability = self.app._compute_recipe_availability(
            ["frozen cherries", "neutral oil", "kosher salt", "unsalted butter", "cooking spray"],
            {
                "kitchen_version": 11,
                "confirmed_staples": ["neutral oil", "kosher salt", "unsalted butter", "cooking spray"],
                "kitchen_candidates": [
                    {
                        "item_id": "item-tomatoes",
                        "display_name": "cherry tomatoes",
                        "tokens": ["cherry", "tomato"],
                    },
                ],
            },
        )

        self.assertFalse(availability["can_make_exact"])
        self.assertEqual(availability["matched_count"], 4)
        self.assertEqual(availability["missing_count"], 1)
        self.assertEqual(availability["missing_ingredients"], ["frozen cherries"])
        self.assertEqual(
            [item["match_status"] for item in availability["ingredient_matches"]],
            ["missing", "pantry", "pantry", "pantry", "pantry"],
        )

    def test_compute_saved_recipe_availability_batch_calls_llm_matcher_once(self):
        kitchen_context = {"kitchen_version": 12, "kitchen_candidates": [{"display_name": "apple", "tokens": ["apple"]}]}
        recipes = [
            {"id": "recipe-1", "title": "One", "ingredients": ["salt"]},
            {"id": "recipe-2", "title": "Two", "ingredients": ["water"]},
        ]
        fake_map = {
            "recipe-1": {"kitchen_version": 12, "can_make_exact": True, "can_make_with_subs": False, "matched_count": 1, "missing_count": 0, "ingredient_matches": [], "missing_ingredients": [], "substitution_candidates": [], "substitution_summary": None, "substitution_status": "none", "analysis_status": "ready"},
            "recipe-2": {"kitchen_version": 12, "can_make_exact": True, "can_make_with_subs": False, "matched_count": 1, "missing_count": 0, "ingredient_matches": [], "missing_ingredients": [], "substitution_candidates": [], "substitution_summary": None, "substitution_status": "none", "analysis_status": "ready"},
        }

        with patch.object(self.app.recipe_inventory_llm, "match_recipes_fast", return_value=(fake_map, {})) as match_recipes:
            availability_map = self.app._compute_saved_recipe_availability_batch(recipes, kitchen_context, request_id="req-1")

        self.assertEqual(availability_map, fake_map)
        match_recipes.assert_called_once()

    def test_llm_inventory_matcher_falls_back_when_payload_invalid(self):
        kitchen_context = {
            "kitchen_version": 7,
            "confirmed_staples": ["salt", "unsalted butter"],
            "kitchen_candidates": [{"display_name": "ground turkey", "tokens": ["ground", "turkey"]}],
        }
        fake_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"recipes": []}'))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=Mock(return_value=fake_response)))
        )

        with patch.dict(os.environ, {"ENABLE_LLM_INVENTORY_MATCHING": "true"}, clear=False), patch.object(
            self.app.recipe_inventory_llm, "_openai_client", return_value=fake_client
        ), patch.object(
            self.app, "_suggest_saved_recipe_substitutions", return_value={"substitution_candidates": [], "substitution_summary": None, "substitution_status": "none", "can_make_with_subs": False}
        ):
            availability_map, meta = self.app.recipe_inventory_llm.match_recipes(
                [{"id": "recipe-1", "title": "Bison Chili", "ingredients": ["bison", "salt"]}],
                kitchen_context,
                substitution_callback=self.app._suggest_saved_recipe_substitutions,
            )

        self.assertTrue(meta["used_fallback"])
        self.assertEqual(availability_map["recipe-1"]["missing_ingredients"], ["bison"])
        self.assertEqual(
            [item["match_status"] for item in availability_map["recipe-1"]["ingredient_matches"]],
            ["missing", "pantry"],
        )

    def test_llm_inventory_matcher_forces_pantry_staples_to_pantry(self):
        fake_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "recipes": [
                                    {
                                        "recipe_id": "recipe-1",
                                        "ingredient_matches": [
                                            {
                                                "recipe_ingredient": "salt",
                                                "canonical_ingredient": "salt",
                                                "match_status": "have",
                                                "matched_kitchen_items": [
                                                    {
                                                        "item_id": "item-salt",
                                                        "display_name": "Diamond Crystal kosher salt",
                                                    }
                                                ],
                                                "substitute_kitchen_items": [],
                                                "missing_reason": None,
                                            },
                                            {
                                                "recipe_ingredient": "unsalted butter",
                                                "canonical_ingredient": "unsalted butter",
                                                "match_status": "missing",
                                                "matched_kitchen_items": [],
                                                "substitute_kitchen_items": [],
                                                "missing_reason": "Model guessed missing.",
                                            },
                                        ],
                                        "substitution_candidates": [],
                                        "substitution_summary": None,
                                        "substitution_status": "none",
                                        "can_make_with_subs": False,
                                    }
                                ]
                            }
                        )
                    )
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=Mock(return_value=fake_response)))
        )

        with patch.dict(os.environ, {"ENABLE_LLM_INVENTORY_MATCHING": "true"}, clear=False), patch.object(
            self.app.recipe_inventory_llm, "_openai_client", return_value=fake_client
        ), patch.object(
            self.app, "_suggest_saved_recipe_substitutions", return_value={"substitution_candidates": [], "substitution_summary": None, "substitution_status": "none", "can_make_with_subs": False}
        ):
            availability_map, meta = self.app.recipe_inventory_llm.match_recipes(
                [{"id": "recipe-1", "title": "Pantry Pasta", "ingredients": ["salt", "unsalted butter"]}],
                {"kitchen_version": 7,
            "confirmed_staples": ["salt", "unsalted butter"], "kitchen_candidates": []},
                substitution_callback=self.app._suggest_saved_recipe_substitutions,
            )

        self.assertFalse(meta["used_fallback"])
        self.assertTrue(availability_map["recipe-1"]["can_make_exact"])
        self.assertEqual(
            [item["match_status"] for item in availability_map["recipe-1"]["ingredient_matches"]],
            ["pantry", "pantry"],
        )
        self.assertEqual(
            availability_map["recipe-1"]["ingredient_matches"][0]["matched_kitchen_items"],
            [],
        )

    def test_suggest_saved_recipe_substitutions_returns_structured_result(self):
        fake_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"substitutions":[{"missing_ingredient":"bison","use_instead":"ground turkey","confidence":"high","notes":"Use 1:1."}],"summary":"Use ground turkey instead of bison."}'
                    )
                )
            ]
        )
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=Mock(return_value=fake_response))
            )
        )

        with patch.object(self.app, "_openai_client", return_value=fake_client):
            substitution = self.app._suggest_saved_recipe_substitutions(
                {"title": "Bison Chili", "ingredients": ["bison", "beans"]},
                {"kitchen_candidates": [{"display_name": "ground turkey"}]},
                ["bison"],
            )

        self.assertTrue(substitution["can_make_with_subs"])
        self.assertEqual(substitution["substitution_status"], "ready")
        self.assertEqual(substitution["substitution_candidates"][0]["use_instead"], "ground turkey")
        self.assertIn("ground turkey", substitution["substitution_summary"])


if __name__ == "__main__":
    unittest.main()
