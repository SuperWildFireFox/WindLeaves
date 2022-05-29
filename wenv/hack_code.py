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
