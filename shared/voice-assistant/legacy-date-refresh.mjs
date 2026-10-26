// Only call after final result validation. Never emit arguments, summaries or
// draft model events; these two compatibility events merely refresh old iOS.
export function legacyDateRefreshEvents(result) {
  const saved = (result?.toolEvents || []).some(event =>
    event.toolName === 'update_item_details' && event.result?.ok === true &&
    Object.hasOwn(event.args || event.result?.args || {}, 'bought_date'));
  return saved ? [
    {type: 'tool_start', tool_name: 'update_item_details'},
    {type: 'tool_end', tool_name: 'update_item_details', ok: true}
  ] : [];
}
