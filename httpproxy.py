TARGET_FILE = "odXH9yzdsj.js"
origin_js = 'await s.am.load();'
hack_js = 'await s.am.load();window.gameInst=this;'

hack_js_file_path = "wenv/odXH9yzdsj.js"

REPLACE_MAP = {
    "/wind-game/": "https://"
}


def response(flow):
    if flow.request.url.find("odXH9yzdsj.js") != -1:
        print("start hack")
        with open(hack_js_file_path, "r", encoding="utf-8") as fp:
            js_content = fp.read()
        for k, v in REPLACE_MAP.items():
            js_content = js_content.replace(k, v)
        flow.response.text = js_content
