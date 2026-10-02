import { DynamoDBClient } from "@aws-sdk/client-dynamodb";
import {
  DynamoDBDocumentClient,
  GetCommand,
  QueryCommand,
  TransactWriteCommand,
} from "@aws-sdk/lib-dynamodb";
import { Fault } from "./core.mjs";
export class DynamoStore {
  constructor(table, client) {
    this.table = table;
    this.client =
      client ||
      DynamoDBDocumentClient.from(new DynamoDBClient({ maxAttempts: 3 }), {
        marshallOptions: { removeUndefinedValues: true },
      });
  }
  async get(pk, sk) {
    return (
      (
        await this.client.send(
          new GetCommand({
            TableName: this.table,
            Key: { pk, sk },
            ConsistentRead: true,
          }),
        )
      ).Item || null
    );
  }
  async list(pk, prefix) {
    let start,
      items = [];
    do {
      const r = await this.client.send(
        new QueryCommand({
          TableName: this.table,
          KeyConditionExpression: "pk = :p AND begins_with(sk, :s)",
          ExpressionAttributeValues: { ":p": pk, ":s": prefix },
          ConsistentRead: true,
          ExclusiveStartKey: start,
        }),
      );
      items.push(...(r.Items || []));
      start = r.LastEvaluatedKey;
    } while (start);
    return items;
  }
  async transaction(changes) {
    try {
      await this.client.send(
        new TransactWriteCommand({
          TransactItems: changes.map(({ item, expected }) => ({
            Put: {
              TableName: this.table,
              Item: item,
              ConditionExpression:
                expected === null ? "attribute_not_exists(pk)" : "#v = :v",
              ...(expected === null
                ? {}
                : {
                    ExpressionAttributeNames: { "#v": "version" },
                    ExpressionAttributeValues: { ":v": expected },
                  }),
            },
          })),
        }),
      );
    } catch (e) {
      if (
        [
          "ConditionalCheckFailedException",
          "TransactionCanceledException",
        ].includes(e.name)
      )
        throw new Fault(
          "conflict",
          "This changed while you were working. Refresh and try again.",
          409,
        );
      throw e;
    }
  }
  async put(pk, sk, data, expected = null) {
    const item = record(pk, sk, data, expected);
    await this.transaction([{ item, expected }]);
    return item;
  }
}
export function record(pk, sk, data, version = null) {
  const item = {
    ...data,
    pk,
    sk,
    version: (version ?? 0) + 1,
    expires: data.expires ?? Math.floor(Date.now() / 1000) + 90 * 86400,
  };
  // Persist the same JSON representation that tools and fingerprinting see.
  // MySQL dates otherwise reach DynamoDB as unsupported class instances.
  const serialized = JSON.stringify(item);
  if (Buffer.byteLength(serialized) > 330000)
    throw new Fault(
      "too_large",
      "This record is too large. Start a new conversation.",
      413,
    );
  return JSON.parse(serialized);
}
export class MemoryStore {
  constructor() {
    this.items = new Map();
  }
  async get(pk, sk) {
    return structuredClone(this.items.get(pk + "\0" + sk) || null);
  }
  async list(pk, prefix) {
    return structuredClone(
      [...this.items.values()]
        .filter((x) => x.pk === pk && x.sk.startsWith(prefix))
        .sort((a, b) => a.sk.localeCompare(b.sk)),
    );
  }
  async transaction(changes) {
    for (const { item, expected } of changes) {
      const old = this.items.get(item.pk + "\0" + item.sk);
      if (expected === null ? !!old : old?.version !== expected)
        throw new Fault(
          "conflict",
          "This changed. Refresh and try again.",
          409,
        );
    }
    for (const { item } of changes)
      this.items.set(item.pk + "\0" + item.sk, structuredClone(item));
  }
  async put(pk, sk, data, expected = null) {
    const item = record(pk, sk, data, expected);
    await this.transaction([{ item, expected }]);
    return item;
  }
}
