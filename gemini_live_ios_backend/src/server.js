import Fastify from 'fastify';
import fastifyWebsocket from '@fastify/websocket';
import fastifyStatic from '@fastify/static';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { BridgeSession } from './bridgeSession.js';
import { loadConfig } from './config.js';
import { GroceryIdentifierClient } from './groceryIdentifierClient.js';
import { KitchenApiClient } from './kitchenApiClient.js';
import { parseInboundMessage } from './protocol.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

function isOriginAllowed(origin, allowedOrigins) {
  if (!origin || allowedOrigins.includes('*')) {
    return true;
  }

  return allowedOrigins.includes(origin);
}

export async function createServer(overrides = {}) {
  const config = {
    ...loadConfig(),
    ...overrides,
  };

  const app = Fastify({
    logger: true,
  });

  await app.register(fastifyWebsocket);
  await app.register(fastifyStatic, {
    root: path.resolve(__dirname, '../public'),
    prefix: '/',
    index: ['index.html'],
  });

  const kitchenApiClient = new KitchenApiClient({
    baseUrl: config.kitchenApiBaseUrl,
    token: config.kitchenApiToken,
    requestTimeoutMs: config.requestTimeoutMs,
    fetchImpl: config.fetchImpl ?? fetch,
  });
  const groceryIdentifierClient = new GroceryIdentifierClient({
    baseUrl: config.groceryIdentifierApiBaseUrl,
    requestTimeoutMs: config.requestTimeoutMs,
    fetchImpl: config.fetchImpl ?? fetch,
  });

  app.get('/health', async () => ({
    status: 'ok',
    geminiModel: config.geminiModel,
    kitchenApiConfigured: kitchenApiClient.isConfigured(),
  }));

  app.get('/ws', { websocket: true }, (socket, request) => {
    if (!isOriginAllowed(request.headers.origin, config.allowedOrigins)) {
      socket.send(
        JSON.stringify({
          type: 'error',
          code: 'origin_not_allowed',
          message: `Origin ${request.headers.origin} is not allowed.`,
        }),
      );
      socket.close();
      return;
    }

    let session = null;

    socket.on('message', async (raw) => {
      try {
        const message = parseInboundMessage(raw.toString());

        if (message.type === 'session.start') {
          session?.close();
          session = new BridgeSession({
            clientSocket: socket,
            config,
            groceryIdentifierClient,
            kitchenApiClient,
            logger: app.log,
            websocketImpl: config.websocketImpl,
          });
          await session.start(message.session);
          return;
        }

        if (!session) {
          socket.send(
            JSON.stringify({
              type: 'error',
              code: 'session_not_started',
              message: 'Send session.start before any other message.',
            }),
          );
          return;
        }

        await session.handleInboundMessage(message);
      } catch (error) {
        // Structured backend-error marker (parity with the Lambda observability
        // fleet). Repo-only for now — ships with the next container deploy.
        try {
          app.log.error(JSON.stringify({
            evt: 'backend_error',
            service: 'voice',
            op: 'gemini_live_ws',
            owner_id: (session && session.ownerId) || null,
            code: 'ws_message_error',
            error: String((error && (error.message || error)) || 'ws_message_error').slice(0, 500),
            job_id: null,
          }));
        } catch (_) { /* never let logging throw */ }
        socket.send(
          JSON.stringify({
            type: 'error',
            code: 'bad_request',
            message: error instanceof Error ? error.message : 'Invalid websocket message.',
          }),
        );
      }
    });

    socket.on('close', () => {
      session?.close();
      session = null;
    });
  });

  return app;
}
