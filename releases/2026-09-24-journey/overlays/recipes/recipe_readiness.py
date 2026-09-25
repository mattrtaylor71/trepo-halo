"""Usable recipe content and retained-link outcomes, independent of job transport."""
import json
import re

INCOMPLETE_MESSAGE = "Your link is saved, but we couldn't get the full recipe. Open it in Saved Recipes to add the ingredients and steps."


def usable_lines(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return []
    if not isinstance(value, list):
        return []
    return [line.strip() for line in value if isinstance(line, str) and line.strip()
            and not re.fullmatch(r'https?://\S+', line.strip(), flags=re.I)
            and line.strip().lower() not in {'not enough recipe information.', 'ingredients', 'instructions', 'steps'}]


def complete(recipe):
    return bool(usable_lines((recipe or {}).get('ingredients'))
                and usable_lines((recipe or {}).get('instructions', (recipe or {}).get('steps'))))


def outcome(recipe):
    if complete(recipe):
        return 'ready'
    if (recipe or {}).get('status') in ('processing', 'repairing'):
        return 'processing'
    return 'link_retained'


def recovery(recipe):
    return {'recipe_id': (recipe or {}).get('id') or (recipe or {}).get('_id'),
            'actions': ['replace_source', 'paste_recipe', 'open_original']}


def incomplete_error(recipe):
    return {'code': 'recipe_content_incomplete', 'recipe_id': recovery(recipe)['recipe_id'],
            'error': INCOMPLETE_MESSAGE, 'retryable': False}
