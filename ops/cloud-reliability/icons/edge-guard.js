// Only immutable generic artwork is admitted to this public cache.
function handler(event) {
  var request = event.request;
  var valid = /^\/icons\/(trepo-grocery-v1|trepo-bold-outline-pilot-v1|trepo-app-bold-outline-v1)\/[0-9a-f]{64}\/[0-9a-f]{64}\.png$/;
  if ((request.method !== 'GET' && request.method !== 'HEAD') || !valid.test(request.uri) || Object.keys(request.querystring || {}).length) {
    return {statusCode:404,statusDescription:'Not Found',headers:{'cache-control':{value:'no-store'}}};
  }
  return request;
}
