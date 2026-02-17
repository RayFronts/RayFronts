"""HTTP Query Service for RayFronts.

Provides a REST API for synchronous semantic queries.

Endpoints:
  POST /query  - Submit a text query, returns position and score
  GET /status  - Health check
"""

import threading
import logging
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from functools import partial

from rayfronts.messaging_services import MessagingService

logger = logging.getLogger(__name__)


class HTTPQueryService(MessagingService):
    """HTTP-based messaging service for synchronous queries."""

    def __init__(self,
                 host: str = "0.0.0.0",
                 port: int = 8080,
                 sync_query_callback=None,
                 text_query_callback=None,
                 debug_callback=None):
        """Initialize HTTP query service.
        
        Args:
            host: Host to bind the HTTP server to.
            port: Port to listen on.
            sync_query_callback: Callback for synchronous queries.
                Should accept (query_text: str) and return dict with
                {position: [x,y,z], score: float, found: bool}.
            text_query_callback: Legacy async callback (unused here).
            debug_callback: Optional callback returning calibration debug info.
        """
        super().__init__()
        self.host = host
        self.port = port
        self.sync_query_callback = sync_query_callback
        self.text_query_callback = text_query_callback
        self.debug_callback = debug_callback
        
        # Create HTTP server with our handler
        handler = partial(_QueryHandler, query_service=self)
        self._server = HTTPServer((host, port), handler)
        
        # Run server in background thread
        self._server_thread = threading.Thread(
            target=self._run_server,
            name="rayfronts_http_server",
            daemon=True)
        self._server_thread.start()
        
        logger.info(f"HTTP Query Service started at http://{host}:{port}")
        logger.info(f"  POST /query  - Submit query")
        logger.info(f"  GET  /status - Health check")
        logger.info(f"  GET  /debug  - Calibration & pose debug info")

    def _run_server(self):
        """Run the HTTP server (blocking)."""
        try:
            self._server.serve_forever()
        except Exception as e:
            logger.error(f"HTTP server error: {e}")

    def process_query(self, query_text: str) -> dict:
        """Process a query synchronously.
        
        Args:
            query_text: The text query string.
            
        Returns:
            Dict with query results or error.
        """
        if self.sync_query_callback is None:
            return {"error": "Query callback not configured", "found": False}
        
        try:
            result = self.sync_query_callback(query_text)
            return result
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            return {"error": str(e), "found": False}

    # MessagingService interface methods
    def text_query_handler(self, s):
        """Handle text query (async, unused in HTTP mode)."""
        if self.text_query_callback is not None:
            self.text_query_callback(s)

    def broadcast_gps_message(self, lat, long):
        raise NotImplementedError()

    def broadcast_map_update(self, map_update):
        raise NotImplementedError()

    def join(self, timeout=None):
        self._server_thread.join(timeout)

    def shutdown(self):
        self._server.shutdown()


class _QueryHandler(BaseHTTPRequestHandler):
    """HTTP request handler for query endpoints."""

    def __init__(self, *args, query_service=None, **kwargs):
        self.query_service = query_service
        super().__init__(*args, **kwargs)

    def log_message(self, format, *args):
        """Override to use our logger."""
        logger.debug(f"HTTP: {args[0]}")

    def _send_json(self, data: dict, status: int = 200):
        """Send JSON response."""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/status":
            self._send_json({"status": "ok", "service": "rayfronts"})
        elif self.path == "/debug":
            if self.query_service.debug_callback:
                self._send_json(self.query_service.debug_callback())
            else:
                self._send_json({"error": "Debug callback not configured"}, 501)
        else:
            self._send_json({"error": "Not found"}, 404)

    def do_POST(self):
        """Handle POST requests."""
        if self.path == "/query":
            self._handle_query()
        else:
            self._send_json({"error": "Not found"}, 404)

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _handle_query(self):
        """Process a query request."""
        try:
            # Read request body
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode()
            
            # Parse JSON
            data = json.loads(body) if body else {}
            query_text = data.get("text", "").strip()
            
            if not query_text:
                self._send_json({"error": "Missing 'text' field"}, 400)
                return
            
            # Process query
            logger.info(f"HTTP query received: '{query_text}'")
            result = self.query_service.process_query(query_text)
            self._send_json(result)
            
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON"}, 400)
        except Exception as e:
            logger.error(f"Query handler error: {e}")
            self._send_json({"error": str(e)}, 500)
