// Test-only substitute for Lambda's injected adapter. Never bundled for deployment.
import {AsyncLocalStorage} from 'node:async_hooks';
export default {actorContext:new AsyncLocalStorage(),requireConsent:async()=>{},providerFetch:f=>f,providerOptions:o=>o};
