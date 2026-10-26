"""Shared value-bound expiration metadata reader. Generated service copies must match."""
from datetime import date, datetime
import re

def expiration_provenance(saved, value):
 """Only expose evidence bound to this exact current date (or explicit clear)."""
 proof=saved.get('__expiration_provenance') if isinstance(saved,dict) else None
 if not isinstance(proof,dict) or type(proof.get('version')) is not int or proof['version']!=1 or proof.get('source') not in ('item_edit','create_request','expiration_image'):return None
 if proof['source']=='expiration_image' and (proof.get('value') is None or not isinstance(proof.get('source_sha256'),str) or not re.fullmatch(r'[a-f0-9]{64}',proof['source_sha256'])):return None
 current=value.isoformat() if type(value) is date else value
 if current is not None:
  if not isinstance(current,str):return None
  try:
   if datetime.strptime(current,'%Y-%m-%d').strftime('%Y-%m-%d')!=current:return None
  except ValueError:return None
 if 'value' not in proof or proof['value']!=current:return None
 try:
  observed=datetime.fromisoformat(proof['observed_at'])
  if observed.tzinfo is None:return None
 except (KeyError,TypeError,ValueError):return None
 return {key:proof[key] for key in ('version','source','value','observed_at')}
