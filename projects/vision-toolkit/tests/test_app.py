import gradio as gr

import app


def test_demo_builds_with_both_tabs():
    demo = app.build_demo()
    assert isinstance(demo, gr.Blocks)


def test_classify_requires_an_image():
    try:
        app.classify_waste(None)
    except gr.Error:
        return
    raise AssertionError("expected gr.Error when no image is given")
