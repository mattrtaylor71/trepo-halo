# Icon CDN deployment access

Status: blocked by AWS authorization, checked October 24, 2026 UTC.

The current `matt-cli` user can read CloudFront resources but cannot create the
function or cache policy in `ops/cloud-reliability/icons/template.json`. The
previous stack `trepo-immutable-icon-edge-20261023` is `ROLLBACK_COMPLETE`.
Its events explicitly report `AccessDenied` for `CreateFunction` and
`CreateCachePolicy`. The attached policy grants CloudFront `Get*` and `List*`;
it does not grant CloudFront writes. No IAM policy was changed.

An AWS administrator can deploy the reviewed template using an already-approved
deployment identity, or provide a narrowly scoped deployment grant. The template
creates only a path-filtering CloudFront Function, a cache policy and a
distribution. It retains the existing hash-validating API origin and private S3
bucket. Do not remove the path guard or expose a bucket to avoid the access block.

For a grant, authorize the CloudFormation resource lifecycle, including rollback:

- `CreateFunction`; `DescribeFunction`, `GetFunction`, `PublishFunction`,
  `UpdateFunction` and `DeleteFunction` for this stack's named function.
- `CreateCachePolicy`; `GetCachePolicy`, `GetCachePolicyConfig`,
  `UpdateCachePolicy` and `DeleteCachePolicy` for the created policy.
- `CreateDistribution`, `TagResource`; `GetDistribution`,
  `GetDistributionConfig`, `UpdateDistribution`, `DeleteDistribution`,
  `ListTagsForResource` and `UntagResource` for the created distribution.

Use account/resource restrictions where supported and scope lifecycle permissions
to the resulting IDs. Creation actions lack a resource ARN before creation;
any broader creation grant should be temporary. `CreateDistributionWithTags` is
an API operation, not a separate IAM action: it requires `CreateDistribution`
and `TagResource`. See the [AWS service authorization reference](https://docs.aws.amazon.com/service-authorization/latest/reference/list_cloudfront.html)
and [tagged distribution API](https://docs.aws.amazon.com/cloudfront/latest/APIReference/API_CreateDistributionWithTags.html).

After access is available, deploy the template with the existing icon API host,
wait for distribution readiness, then verify repeated valid image GETs (identical
bytes and an actual cache hit), HEAD, invalid hash/style/path, query strings and
non-GET methods. Only after those checks should the resolver's `ICON_CDN_HOST`
be enabled using its current revision. Preserve its previous configuration for
rollback. No CDN address is currently enabled.

The existing origin capacity increase from 10 to 30 is live and independently
passed a 40-request burst. That mitigation does not establish edge-cache behavior.
