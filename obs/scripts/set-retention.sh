#!/usr/bin/env bash
#
# set-retention.sh — Set 90-day retention on ALL /aws/lambda/* CloudWatch log groups.
#
# =====================================================================================
#  ⚠️  DESTRUCTIVE — READ BEFORE RUNNING  ⚠️
# =====================================================================================
#  CloudWatch applies a retention policy by DELETING log events older than the
#  retention window. Many Trepo Lambda log groups are currently "Never expire".
#  Setting them to 90 days will PERMANENTLY DELETE all log data older than 90 days
#  the next time CloudWatch enforces retention. This CANNOT be undone.
#
#  Requires Matt's explicit consent. WP1 did NOT run this script.
#
#  This script refuses to do anything unless invoked with --confirm.
# =====================================================================================
#
# Usage (only after consent):
#   AWS_PROFILE=trepo-dev AWS_REGION=us-east-1 ./set-retention.sh --confirm
#
# Dry run (safe — lists affected log groups and their current retention, changes nothing):
#   AWS_PROFILE=trepo-dev AWS_REGION=us-east-1 ./set-retention.sh --dry-run
set -euo pipefail

export AWS_PROFILE="${AWS_PROFILE:-trepo-dev}"
export AWS_REGION="${AWS_REGION:-us-east-1}"
RETENTION_DAYS=90
PREFIX="/aws/lambda/"

MODE="${1:-}"

if [[ "$MODE" != "--confirm" && "$MODE" != "--dry-run" ]]; then
  cat >&2 <<'EOF'
REFUSING TO RUN.

This script permanently deletes CloudWatch Lambda logs older than 90 days.
It requires an explicit flag:

  --dry-run   List every /aws/lambda/* log group and its current retention. Changes nothing.
  --confirm   Actually apply 90-day retention to every /aws/lambda/* log group. DESTRUCTIVE.

Re-run with one of those flags. Do not run --confirm without Matt's consent.
EOF
  exit 2
fi

echo "Enumerating log groups under ${PREFIX} ..."
mapfile -t GROUPS < <(
  aws logs describe-log-groups --log-group-name-prefix "$PREFIX" \
    --query 'logGroups[].logGroupName' --output text | tr '\t' '\n' | sed '/^$/d'
)
echo "Found ${#GROUPS[@]} log group(s)."

if [[ "$MODE" == "--dry-run" ]]; then
  echo "DRY RUN — current retention (days; 'none' = never expire):"
  aws logs describe-log-groups --log-group-name-prefix "$PREFIX" \
    --query 'logGroups[].[logGroupName, (retentionInDays || `none`)]' --output text
  echo "DRY RUN complete. No changes made. Would set all of the above to ${RETENTION_DAYS} days."
  exit 0
fi

# --confirm path (destructive)
echo ""
echo "############################################################"
echo "# DESTRUCTIVE: applying ${RETENTION_DAYS}-day retention to"
echo "# ${#GROUPS[@]} log group(s). Logs older than ${RETENTION_DAYS}d will be deleted."
echo "############################################################"
read -r -p "Type EXACTLY 'DELETE OLD LOGS' to proceed: " ACK
if [[ "$ACK" != "DELETE OLD LOGS" ]]; then
  echo "Confirmation phrase not matched. Aborting. No changes made."
  exit 3
fi

for g in "${GROUPS[@]}"; do
  aws logs put-retention-policy --log-group-name "$g" --retention-in-days "$RETENTION_DAYS"
  echo "  set ${RETENTION_DAYS}d: $g"
done
echo "Done. Applied ${RETENTION_DAYS}-day retention to ${#GROUPS[@]} log group(s)."
