from http.server import BaseHTTPRequestHandler, HTTPServer
import json

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/data":
            with open("output.json") as f:
                data = json.load(f)

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())

server = HTTPServer(("localhost", 8000), Handler)
print("🚀 http://localhost:8000/data")
server.serve_forever()
