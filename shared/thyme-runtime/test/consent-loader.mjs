export async function resolve(specifier,context,next) {
  if(specifier==='/var/task/ai_consent_guard.cjs')return {url:new URL('./consent-fixture.mjs',import.meta.url).href,shortCircuit:true};
  return next(specifier,context);
}
