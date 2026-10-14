import http.server
import socketserver
import webbrowser
import os
import sys

PORT = 8080
DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web')

class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

def run():
    # Allow address reuse
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), CustomHandler) as httpd:
        url = f"http://localhost:{PORT}"
        print("=" * 60)
        print(f" 50 Startups ML Regression Web Dashboard 啟動成功！")
        print(f" 本地網址 (URL): {url}")
        print(f" 網頁根目錄: {DIRECTORY}")
        print("=" * 60)
        print(" 按 Ctrl + C 即可停止伺服器。\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[+] 伺服器已正常停止。")

if __name__ == '__main__':
    run()
