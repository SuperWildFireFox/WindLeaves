def get_keydown_js(keycode):
    assert isinstance(keycode, str) and len(keycode) == 1
    js_code = """
hack_game_area[0].dispatchEvent(new KeyboardEvent("keydown", {{
    key: "{}",
    keyCode: 69,        // example values.
    code: "Key{}",       // put everything you need in this object.
    which: 69,
    shiftKey: false,    // you don't need to include values
    ctrlKey: false,     // if you aren't going to use them.
    metaKey: false      // these are here for example's sake.
}}));
    """.format(keycode.lower(), keycode.upper())
    return js_code


def get_keyup_js(keycode):
    assert isinstance(keycode, str) and len(keycode) == 1
    js_code = """
hack_game_area[0].dispatchEvent(new KeyboardEvent("keyup", {{
    key: "{}",
    keyCode: 69,        // example values.
    code: "Key{}",       // put everything you need in this object.
    which: 69,
    shiftKey: false,    // you don't need to include values
    ctrlKey: false,     // if you aren't going to use them.
    metaKey: false      // these are here for example's sake.
}}));
    """.format(keycode.lower(), keycode.upper())
    return js_code


# fround https://www.bilibili.com/video/BV1ma41147a2
refound_js_code = """
var myGameScript = document.createElement("script");
myGameScript.setAttribute("type","text/javascript");
myGameScript.setAttribute("src","https://activity.hdslb.com/blackboard/static/20220330/00979505aec5edd6e5c2f8c096fa0f62/odXH9yzdsj.js")
myGameScript.addEventListener('load', (event) => {
var myContainer = document.getElementById("i_cecream");
myContainer.innerHTML="";
var gameInst = new BannerGameSpring2022(myContainer);
gameInst.init().then( ()=>gameInst.start() );
});
document.body.appendChild(myGameScript);
"""