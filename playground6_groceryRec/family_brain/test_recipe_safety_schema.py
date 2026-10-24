import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('recipe_safety_schema',Path(__file__).with_name('recipe_safety.py'))
safety=importlib.util.module_from_spec(spec);spec.loader.exec_module(safety)

@pytest.mark.parametrize('field',['instructions','steps'])
@pytest.mark.parametrize('allergy,food',[('peanuts','peanut butter'),('milk','heavy cream'),('eggs','egg'),('soy','soy sauce'),('sesame','sesame oil')])
def test_instruction_only_allergen_is_checked_in_each_generated_schema(field,allergy,food):
    recipe={'title':'Rice bowl','ingredients':['rice','water'],field:[f'Stir in {food}.']}
    assert safety._recipe_violates(recipe,{'allergies':[allergy]}) is not None

def test_both_fields_are_checked_when_provider_emits_both():
    recipe={'title':'Rice','ingredients':['rice'],'instructions':['Boil rice.'],'steps':['Mix in peanut butter.']}
    assert safety._recipe_violates(recipe,{'allergies':['peanuts']}) is not None

@pytest.mark.parametrize('field',['instructions','steps'])
def test_qualified_plant_substitute_stays_allowed(field):
    recipe={'title':'Vegan rice','ingredients':['rice','vegan butter'],field:['Spread butter over the rice.']}
    assert safety._recipe_violates(recipe,{'allergies':['milk']}) is None

@pytest.mark.parametrize('value',[None,[],{},3,'Boil rice.'])
def test_legacy_empty_or_text_schema_does_not_crash(value):
    assert safety._recipe_violates({'title':'Rice','ingredients':['rice'],'steps':value},{'allergies':['peanuts']}) is None
